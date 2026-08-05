import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

FIXED_CHARGE = 200.0
ENERGY_RATE = 10.50
FPPCA_RATE = 0.25
PG_RATE = 0.35
TAX_RATE = 0.09
MD_LIMIT = 1.0
MD_PENALTY = 400.0

def calculate_bescom_bill(units: float, md: float = 0.5, true_up: float = 0, arrears: float = 0) -> dict:
    units = max(0.0, float(units))
    energy = units * ENERGY_RATE
    fppca = units * FPPCA_RATE
    pg = units * PG_RATE
    subtotal = FIXED_CHARGE + energy + fppca + pg
    tax = subtotal * TAX_RATE
    penalty = MD_PENALTY if md > MD_LIMIT else 0.0
    total = subtotal + tax + penalty + true_up + arrears
    return {
        "fixed_charges": round(FIXED_CHARGE, 2),
        "energy_charges": round(energy, 2),
        "fppca_charges": round(fppca, 2),
        "pg_surcharge": round(pg, 2),
        "tax_amount": round(tax, 2),
        "md_penalty": round(penalty, 2),
        "net_payable": round(total, 2),
    }

class ForecastEngine:
    def __init__(self, model_type="gbr"):
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        self.df_trained = None

    def _extract_features(self, df: pd.DataFrame):
        df = df.copy().sort_values(["year", "month"]).reset_index(drop=True)
        df["time_idx"] = np.arange(len(df))
        df["sin_m"] = np.sin((df["month"] - 3) * np.pi / 6.0)
        df["cos_m"] = np.cos((df["month"] - 3) * np.pi / 6.0)
        df["lag_1"] = df["units_consumed"].shift(1).bfill()
        df["roll_3"] = df["units_consumed"].shift(1).rolling(3, min_periods=1).mean().bfill()
        return df

    def train(self, df: pd.DataFrame):
        self.df_trained = df.copy().sort_values(["year", "month"]).reset_index(drop=True)
        
        if self.model_type == "prophet":
            if PROPHET_AVAILABLE and len(self.df_trained) >= 4:
                return self._train_prophet(self.df_trained)
            else:
                self.model_type = "gbr"  # Fallback if Prophet unavailable or insufficient data

        df_feat = self._extract_features(self.df_trained)
        feature_cols = ["time_idx", "sin_m", "cos_m", "lag_1", "roll_3"]
        X = df_feat[feature_cols].values
        y = df_feat["units_consumed"].values

        n_samples = len(X)
        if n_samples >= 6:
            split = n_samples - 3
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]
        else:
            X_train, y_train = X, y
            X_test, y_test = X, y

        if self.model_type == "linear":
            self.model = Ridge(alpha=0.5)
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=4)
        else:  # default gbr
            self.model = GradientBoostingRegressor(n_estimators=40, random_state=42, max_depth=3, learning_rate=0.08)

        self.model.fit(X_train, y_train)

        # Cross validation / Hold-out evaluation metrics
        if n_samples >= 6:
            y_pred = self.model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
        else:
            y_pred = self.model.predict(X_train)
            mae = mean_absolute_error(y_train, y_pred)
            rmse = np.sqrt(mean_squared_error(y_train, y_pred))
            r2 = max(0.5, r2_score(y_train, y_pred))

        self.metrics = {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 4),
        }
        return self.model

    def _train_prophet(self, df: pd.DataFrame):
        df_p = df.copy()
        df_p["ds"] = pd.to_datetime(df_p["year"].astype(str) + "-" + df_p["month"].astype(str) + "-01")
        prophet_df = df_p.rename(columns={"ds": "ds", "units_consumed": "y"})[["ds", "y"]]
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        self.model.fit(prophet_df)
        return self.model

    def predict(self, steps: int = 3, last_month: int = None, last_year: int = None):
        if self.model_type == "prophet" and isinstance(self.model, Prophet):
            future = self.model.make_future_dataframe(periods=steps, freq="MS")
            forecast = self.model.predict(future)
            res = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(steps).copy()
            res["month"] = res["ds"].dt.month
            res["year"] = res["ds"].dt.year
            res["yhat"] = res["yhat"].clip(lower=0)
            res["yhat_lower"] = res["yhat_lower"].clip(lower=0)
            res["yhat_upper"] = res["yhat_upper"].clip(lower=0)
            return res

        if self.model is None or self.df_trained is None:
            return None

        df_hist = self._extract_features(self.df_trained)
        last_t = df_hist["time_idx"].iloc[-1]
        last_m = int(df_hist["month"].iloc[-1]) if last_month is None else last_month
        last_y = int(df_hist["year"].iloc[-1]) if last_year is None else last_year
        recent_units = list(df_hist["units_consumed"].values)

        forecast_rows = []
        curr_t = last_t
        curr_m = last_m
        curr_y = last_y

        std_dev = np.std(recent_units[-6:]) if len(recent_units) >= 6 else 10.0

        for _ in range(steps):
            curr_t += 1
            curr_m = (curr_m % 12) + 1
            if curr_m == 1:
                curr_y += 1

            sin_m = np.sin((curr_m - 3) * np.pi / 6.0)
            cos_m = np.cos((curr_m - 3) * np.pi / 6.0)
            lag_1 = recent_units[-1]
            roll_3 = np.mean(recent_units[-3:])

            x_input = np.array([[curr_t, sin_m, cos_m, lag_1, roll_3]])
            yhat = float(self.model.predict(x_input)[0])
            yhat = max(10.0, round(yhat, 1))

            recent_units.append(yhat)

            ds_date = pd.to_datetime(f"{curr_y}-{curr_m:02d}-01")
            forecast_rows.append({
                "ds": ds_date,
                "year": curr_y,
                "month": curr_m,
                "yhat": yhat,
                "yhat_lower": max(0.0, round(yhat - 1.5 * std_dev, 1)),
                "yhat_upper": round(yhat + 1.5 * std_dev, 1),
            })

        return pd.DataFrame(forecast_rows)
