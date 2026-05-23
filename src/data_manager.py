import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "bill_history.csv"

COLUMNS = [
    "bill_period", "month", "year",
    "present_reading", "previous_reading", "units_consumed",
    "recorded_md", "power_factor",
    "fixed_charges", "energy_charges", "fppca_charges", "pg_surcharge",
    "tax_amount", "md_penalty", "true_up_charges", "arrears",
    "net_payable", "source"
]

def load_history() -> pd.DataFrame:
    if not DATA_PATH.exists() or DATA_PATH.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(DATA_PATH)
    df["month"] = df["month"].astype(int)
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df

def save_entry(entry: dict):
    df = load_history()
    key_m, key_y = entry.get("month"), entry.get("year")
    mask = (df["month"] == key_m) & (df["year"] == key_y)
    if mask.any():
        df = df[~mask]
    new_row = pd.DataFrame([{k: entry.get(k, 0) for k in COLUMNS}])
    df = pd.concat([df, new_row], ignore_index=True)
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    df.to_csv(DATA_PATH, index=False)

def delete_entry(month: int, year: int):
    df = load_history()
    df = df[~((df["month"] == month) & (df["year"] == year))]
    df.to_csv(DATA_PATH, index=False)

def update_entry(month: int, year: int, updates: dict):
    df = load_history()
    mask = (df["month"] == month) & (df["year"] == year)
    for k, v in updates.items():
        if k in COLUMNS:
            df.loc[mask, k] = v
    df.to_csv(DATA_PATH, index=False)
