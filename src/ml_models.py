import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from prophet import Prophet

FIXED_CHARGE = 200.0
ENERGY_RATE = 10.50
FPPCA_RATE = 0.25
PG_RATE = 0.35
TAX_RATE = 0.09
MD_LIMIT = 1.0
MD_PENALTY = 400.0

def calculate_bescom_bill(units: float, md: float = 0.5, true_up: float = 0, arrears: float = 0) -> dict:
    energy = units * ENERGY_RATE
    fppca = units * FPPCA_RATE
    pg = units * PG_RATE
    subtotal = FIXED_CHARGE + energy + fppca + pg
    tax = subtotal * TAX_RATE
    penalty = MD_PENALTY if md > MD_LIMIT else 0
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
    def __init__(self, model_type="linear"):
        self.model_type = model_type
        self.model = None
        self.metrics = {}
        self._X_train = None
        self._y_train = None

    def train(self, df: pd.DataFrame):
        if self.model_type == "prophet":
            return self._train_prophet(df)
        df = df.sort_values(["year", "month"]).reset_index(drop=True)
        df["time_idx"] = range(len(df))
        self._features = ["time_idx"]
        if "month" in df.columns:
            self._features.append("month")
        X = df[self._features].values
        y = df["units_consumed"].values
        if len(X) < 4:
            X_train, y_train = X, y
            X_test, y_test = X, y
        else:
            split = max(2, int(len(X) * 0.8))
            X_train, y_train = X[:split], y[:split]
            X_test, y_test = X[split:], y[split:]
        self._X_train, self._y_train = X_train, y_train
        if self.model_type == "linear":
            self.model = LinearRegression()
        else:
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        self.metrics = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "R2": round(r2_score(y_test, y_pred), 4),
        }
        return self.model

    def _train_prophet(self, df: pd.DataFrame):
        df = df.copy().sort_values(["year", "month"])
        df["ds"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
        prophet_df = df.rename(columns={"ds": "ds", "units_consumed": "y"})
        self.model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        self.model.fit(prophet_df[["ds", "y"]])
        return self.model

    def predict(self, steps: int = 3, last_month: int = None, last_year: int = None):
        if self.model_type == "prophet" and isinstance(self.model, Prophet):
            future = self.model.make_future_dataframe(periods=steps, freq="MS")
            forecast = self.model.predict(future)
            result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(steps).copy()
            result["month"] = result["ds"].dt.month
            result["year"] = result["ds"].dt.year
            return result
        if self.model is None:
            return None
        last_idx = self._X_train[-1][0] if self._X_train is not None else 0
        has_month = len(self._features) > 1 if hasattr(self, "_features") else False
        preds = []
        for i in range(steps):
            t = last_idx + 1 + i
            if has_month:
                m = ((last_month - 1 + i) % 12) + 1 if last_month else 1
                pred = self.model.predict([[t, m]])[0]
            else:
                pred = self.model.predict([[t]])[0]
            preds.append(max(0, pred))
        return np.array(preds)
