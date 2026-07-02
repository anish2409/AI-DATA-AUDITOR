import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp


def calculate_numeric_psi(baseline_series, current_series, buckets=10):
    baseline = pd.Series(baseline_series).dropna()
    current = pd.Series(current_series).dropna()

    if baseline.empty or current.empty:
        return np.nan

    breakpoints = np.percentile(baseline, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        return np.nan

    baseline_counts = np.histogram(baseline, bins=breakpoints)[0]
    current_counts = np.histogram(current, bins=breakpoints)[0]

    baseline_percents = baseline_counts / max(baseline_counts.sum(), 1)
    current_percents = current_counts / max(current_counts.sum(), 1)

    baseline_percents = np.where(baseline_percents == 0, 0.0001, baseline_percents)
    current_percents = np.where(current_percents == 0, 0.0001, current_percents)

    psi = np.sum(
        (current_percents - baseline_percents)
        * np.log(current_percents / baseline_percents)
    )

    return round(float(psi), 4)


def calculate_categorical_psi(baseline_series, current_series):
    baseline = pd.Series(baseline_series).fillna("MISSING").astype(str)
    current = pd.Series(current_series).fillna("MISSING").astype(str)

    categories = sorted(set(baseline.unique()).union(set(current.unique())))

    baseline_dist = baseline.value_counts(normalize=True).reindex(categories, fill_value=0)
    current_dist = current.value_counts(normalize=True).reindex(categories, fill_value=0)

    baseline_dist = baseline_dist.replace(0, 0.0001)
    current_dist = current_dist.replace(0, 0.0001)

    psi = np.sum(
        (current_dist - baseline_dist)
        * np.log(current_dist / baseline_dist)
    )

    return round(float(psi), 4)


def drift_severity(psi):
    if pd.isna(psi):
        return "Unknown"

    if psi < 0.10:
        return "No Significant Drift"

    if psi < 0.25:
        return "Moderate Drift"

    return "Significant Drift"


def detect_drift(baseline_path, current_path, target_column=None):
    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)

    common_columns = [
        column for column in baseline_df.columns
        if column in current_df.columns
    ]

    results = []

    for column in common_columns:
        baseline_series = baseline_df[column]
        current_series = current_df[column]

        baseline_missing = round(baseline_series.isna().mean() * 100, 2)
        current_missing = round(current_series.isna().mean() * 100, 2)
        missing_drift = round(current_missing - baseline_missing, 2)

        column_role = "target" if column == target_column else "feature"

        if pd.api.types.is_numeric_dtype(baseline_series):
            psi = calculate_numeric_psi(baseline_series, current_series)

            baseline_clean = baseline_series.dropna()
            current_clean = current_series.dropna()

            if len(baseline_clean) > 0 and len(current_clean) > 0:
                ks_stat, ks_pvalue = ks_2samp(baseline_clean, current_clean)
                ks_stat = round(float(ks_stat), 4)
                ks_pvalue = round(float(ks_pvalue), 6)
            else:
                ks_stat = np.nan
                ks_pvalue = np.nan

            results.append({
                "column_name": column,
                "column_role": column_role,
                "data_type": "numeric",
                "baseline_missing_percent": baseline_missing,
                "current_missing_percent": current_missing,
                "missing_drift_percent": missing_drift,
                "baseline_mean": round(baseline_series.mean(skipna=True), 2),
                "current_mean": round(current_series.mean(skipna=True), 2),
                "baseline_top_value": None,
                "current_top_value": None,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pvalue,
                "drift_severity": drift_severity(psi)
            })

        else:
            psi = calculate_categorical_psi(baseline_series, current_series)

            baseline_top = baseline_series.fillna("MISSING").astype(str).mode()
            current_top = current_series.fillna("MISSING").astype(str).mode()

            results.append({
                "column_name": column,
                "column_role": column_role,
                "data_type": "categorical_or_text",
                "baseline_missing_percent": baseline_missing,
                "current_missing_percent": current_missing,
                "missing_drift_percent": missing_drift,
                "baseline_mean": None,
                "current_mean": None,
                "baseline_top_value": baseline_top.iloc[0] if not baseline_top.empty else None,
                "current_top_value": current_top.iloc[0] if not current_top.empty else None,
                "psi": psi,
                "ks_statistic": None,
                "ks_pvalue": None,
                "drift_severity": drift_severity(psi)
            })

    return pd.DataFrame(results)


def generate_drift_report(drift_df, output_path):
    output = []

    output.append("# Data Drift Detection Report")
    output.append("")
    output.append("## Drift Summary")
    output.append("")

    total_columns = len(drift_df)
    significant_count = len(drift_df[drift_df["drift_severity"] == "Significant Drift"])
    moderate_count = len(drift_df[drift_df["drift_severity"] == "Moderate Drift"])

    output.append(f"- Total columns checked: {total_columns}")
    output.append(f"- Moderate drift columns: {moderate_count}")
    output.append(f"- Significant drift columns: {significant_count}")

    output.append("")
    output.append("## Drifted Columns")
    output.append("")

    drifted_columns = drift_df[
        drift_df["drift_severity"].isin(["Moderate Drift", "Significant Drift"])
    ].sort_values(by="psi", ascending=False)

    if drifted_columns.empty:
        output.append("- No moderate or significant drift detected.")
    else:
        for _, row in drifted_columns.iterrows():
            output.append(
                f"- `{row['column_name']}` | {row['drift_severity']} | PSI: {row['psi']}"
            )

    output.append("")
    output.append("## Missing Value Drift")
    output.append("")

    missing_drift = drift_df[
        drift_df["missing_drift_percent"].abs() >= 3
    ].sort_values(by="missing_drift_percent", ascending=False)

    if missing_drift.empty:
        output.append("- No major missing value drift detected.")
    else:
        for _, row in missing_drift.iterrows():
            output.append(
                f"- `{row['column_name']}` missing changed from "
                f"{row['baseline_missing_percent']}% to {row['current_missing_percent']}%"
            )

    output.append("")
    output.append("## Full Drift Table")
    output.append("")
    output.append(drift_df.to_markdown(index=False))

    output.append("")
    output.append("## Interpretation")
    output.append("")
    output.append("- PSI < 0.10 means no significant drift.")
    output.append("- PSI between 0.10 and 0.25 means moderate drift.")
    output.append("- PSI > 0.25 means significant drift.")
    output.append("- Drifted features should be investigated before model retraining or deployment.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    baseline_path = "data/raw/sample_customer_ai_audit_dataset.csv"
    current_path = "data/drift_samples/current_customer_ai_audit_dataset.csv"
    target_column = "churn"

    drift_df = detect_drift(
        baseline_path=baseline_path,
        current_path=current_path,
        target_column=target_column
    )

    Path("data/exports").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    drift_df.to_csv("data/exports/drift_summary.csv", index=False)

    generate_drift_report(
        drift_df,
        "reports/drift_report.md"
    )

    moderate_count = len(
        drift_df[drift_df["drift_severity"] == "Moderate Drift"]
    )

    significant_count = len(
        drift_df[drift_df["drift_severity"] == "Significant Drift"]
    )

    print("Drift detection completed.")
    print(f"Moderate drift columns: {moderate_count}")
    print(f"Significant drift columns: {significant_count}")
    print("Report saved: reports/drift_report.md")
    print("Drift summary saved: data/exports/drift_summary.csv")
    