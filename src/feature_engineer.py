import pandas as pd
import numpy as np

SEASON_MAP = {1:"Winter",2:"Winter",3:"Summer",4:"Summer",5:"Summer",
              6:"Pre-Monsoon",7:"Monsoon",8:"Monsoon",9:"Monsoon",
              10:"Post-Monsoon",11:"Post-Monsoon",12:"Winter"}

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy().sort_values(["year", "month"])
    df["time_idx"] = range(len(df))
    df["season"] = df["month"].map(SEASON_MAP)
    df["season_code"] = df["season"].map({
        "Winter":0,"Summer":1,"Pre-Monsoon":2,"Monsoon":3,"Post-Monsoon":4
    })
    df["units_lag_1"] = df["units_consumed"].shift(1)
    df["units_lag_2"] = df["units_consumed"].shift(2)
    df["units_roll_3"] = df["units_consumed"].rolling(3, min_periods=1).mean()
    df["units_roll_6"] = df["units_consumed"].rolling(6, min_periods=1).mean()
    df["md_lag_1"] = df["recorded_md"].shift(1)
    df["cost_per_unit"] = df["net_payable"] / df["units_consumed"].replace(0, np.nan)
    df["md_exceeded"] = (df["recorded_md"] > 1.0).astype(int)
    return df

def prepare_ml_data(df: pd.DataFrame, target: str = "units_consumed"):
    df = create_features(df).dropna()
    if df.empty:
        return None, None
    exclude = ["bill_period", "source", "season"]
    feature_cols = [c for c in df.columns if c not in exclude and c != target
                    and not c.startswith(("energy_","fppca_","pg_","tax_","md_penalty",
                    "true_up_","arrears","fixed_","net_payable","cost_"))]
    X = df[feature_cols]
    y = df[target]
    return X, y
