import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(99)

BASELINE_PATH = Path("data/raw/sample_customer_ai_audit_dataset.csv")
OUTPUT_DIR = Path("data/drift_samples")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(BASELINE_PATH)

current_df = df.copy()

current_df["income"] = current_df["income"] * np.random.normal(
    loc=0.88,
    scale=0.08,
    size=len(current_df)
)

current_df["monthly_spend"] = current_df["monthly_spend"] * np.random.normal(
    loc=1.22,
    scale=0.12,
    size=len(current_df)
)

current_df["support_tickets"] = current_df["support_tickets"] + np.random.poisson(
    lam=1,
    size=len(current_df)
)

region_shift_index = current_df.sample(frac=0.22, random_state=99).index
current_df.loc[region_shift_index, "region"] = "West"

payment_shift_index = current_df.sample(frac=0.12, random_state=100).index
current_df.loc[payment_shift_index, "last_payment_status"] = "failed"

missing_income_index = current_df.sample(frac=0.05, random_state=101).index
current_df.loc[missing_income_index, "income"] = np.nan

churn_shift_index = current_df.sample(frac=0.08, random_state=102).index
current_df.loc[churn_shift_index, "churn"] = 1

output_path = OUTPUT_DIR / "current_customer_ai_audit_dataset.csv"

current_df.to_csv(output_path, index=False)

print("Current drift sample created.")
print(f"Output path: {output_path}")
print(f"Shape: {current_df.shape}")