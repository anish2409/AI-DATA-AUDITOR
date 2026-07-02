import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

n = 1000

df = pd.DataFrame({
    "customer_id": np.arange(1, n + 1),
    "age": np.random.normal(35, 12, n).round(),
    "income": np.random.normal(50000, 15000, n).round(2),
    "gender": np.random.choice(
        ["Male", "Female", "male", "F", "Unknown"],
        n,
        p=[0.38, 0.38, 0.08, 0.08, 0.08]
    ),
    "region": np.random.choice(
        ["North", "South", "East", "West", "north", "UNKNOWN"],
        n
    ),
    "tenure_months": np.random.randint(0, 72, n),
    "monthly_spend": np.random.normal(2500, 900, n).round(2),
    "support_tickets": np.random.poisson(2, n),
    "last_payment_status": np.random.choice(
        ["paid", "failed", "pending"],
        n,
        p=[0.82, 0.12, 0.06]
    ),
    "churn": np.random.choice([0, 1], n, p=[0.82, 0.18])
})

df.loc[np.random.choice(df.index, 80, replace=False), "income"] = np.nan
df.loc[np.random.choice(df.index, 45, replace=False), "gender"] = np.nan
df.loc[np.random.choice(df.index, 30, replace=False), "monthly_spend"] = np.nan

df.loc[np.random.choice(df.index, 10, replace=False), "age"] = -5
df.loc[np.random.choice(df.index, 8, replace=False), "monthly_spend"] = -100

df.loc[np.random.choice(df.index, 5, replace=False), "income"] = 500000
df.loc[np.random.choice(df.index, 5, replace=False), "monthly_spend"] = 50000

duplicates = df.sample(15, random_state=42)
df = pd.concat([df, duplicates], ignore_index=True)

output_path = OUTPUT_DIR / "sample_customer_ai_audit_dataset.csv"
df.to_csv(output_path, index=False)

print(f"Sample dirty dataset created: {output_path}")
print(f"Shape: {df.shape}")
