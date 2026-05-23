import pandas as pd
import numpy as np
from src.ml_models import calculate_bescom_bill

np.random.seed(42)
rows = []
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

base = 110
for i in range(14):
    m = (4 + i) % 12 or 12
    y = 2025 + (4 + i - 1) // 12
    seasonal = 1 + 0.35 * np.sin((m - 6) * np.pi / 6)
    noise = np.random.normal(1, 0.06)
    md_noise = np.random.normal(0.5, 0.3)
    units = round(base * seasonal * noise, 1)
    md = max(0.2, round(0.5 + md_noise, 3))
    pf = round(min(0.99, max(0.85, 0.95 + np.random.normal(0, 0.02))), 2)
    bill = calculate_bescom_bill(units, md)
    true_up = round(np.random.choice([0, 0, 0, 150, 230, 0, 0, 80, 0, 200, 0, 0, 0, 0], 1)[0], 2)
    arrears = 0.0
    total = round(bill["net_payable"] + true_up + arrears, 2)
    rows.append({
        "bill_period": f"{month_names[m-1]}-{y}",
        "month": m, "year": y,
        "present_reading": 0, "previous_reading": 0,
        "units_consumed": units,
        "recorded_md": md, "power_factor": pf,
        "fixed_charges": bill["fixed_charges"],
        "energy_charges": bill["energy_charges"],
        "fppca_charges": bill["fppca_charges"],
        "pg_surcharge": bill["pg_surcharge"],
        "tax_amount": bill["tax_amount"],
        "md_penalty": bill["md_penalty"],
        "true_up_charges": true_up,
        "arrears": arrears,
        "net_payable": total,
        "source": "seed"
    })

df = pd.DataFrame(rows)
present = 2000 + df["units_consumed"].cumsum()
previous = present.shift(1).fillna(1900)
df["present_reading"] = present.round(1)
df["previous_reading"] = previous.round(1)
df.to_csv("data/bill_history.csv", index=False)
print(f"Seeded {len(df)} records from {rows[0]['bill_period']} to {rows[-1]['bill_period']}")
print(df[["bill_period", "units_consumed", "recorded_md", "net_payable"]].tail(5).to_string(index=False))
