import pandas as pd
import numpy as np
from pathlib import Path
from data_profiler import profile_dataset


SUSPICIOUS_LEAKAGE_TERMS = [
    "last",
    "status",
    "result",
    "outcome",
    "label",
    "target",
    "after",
    "post",
    "final",
    "closed",
    "resolved"
]


SENSITIVE_OR_GROUP_COLUMNS = [
    "gender",
    "region",
    "age_group",
    "state",
    "city",
    "segment"
]


def detect_leakage_risks(df, profile_df, target_column):
    warnings = []

    identifier_columns = profile_df[
        profile_df["likely_identifier_flag"] == True
    ]["column_name"].tolist()

    for column in identifier_columns:
        warnings.append({
            "column_name": column,
            "risk_type": "Identifier Leakage Risk",
            "severity": "High",
            "reason": "Identifier-like columns can cause memorization and should not be used directly as model features."
        })

    for column in df.columns:
        column_lower = column.lower()

        if column == target_column:
            continue

        for term in SUSPICIOUS_LEAKAGE_TERMS:
            if term in column_lower:
                warnings.append({
                    "column_name": column,
                    "risk_type": "Suspicious Column Name",
                    "severity": "Medium",
                    "reason": f"Column name contains '{term}', which may indicate post-event or outcome-related information."
                })
                break

    if target_column in df.columns:
        target = df[target_column]

        for column in df.columns:
            if column == target_column:
                continue

            if pd.api.types.is_numeric_dtype(df[column]):
                valid_data = df[[column, target_column]].dropna()

                if len(valid_data) > 10:
                    corr = valid_data[column].corr(valid_data[target_column])

                    if pd.notna(corr) and abs(corr) >= 0.75:
                        warnings.append({
                            "column_name": column,
                            "risk_type": "High Correlation With Target",
                            "severity": "High",
                            "reason": f"Numeric column has very high correlation with target: {round(corr, 4)}"
                        })

    return pd.DataFrame(warnings)


def detect_bias_and_group_risk(df, target_column):
    fairness_rows = []

    if target_column not in df.columns:
        return pd.DataFrame(fairness_rows)

    for column in df.columns:
        column_lower = column.lower()

        if column_lower not in SENSITIVE_OR_GROUP_COLUMNS:
            continue

        temp = df[[column, target_column]].dropna()

        if temp.empty:
            continue

        group_summary = temp.groupby(column).agg(
            group_size=(target_column, "count"),
            target_rate=(target_column, "mean")
        ).reset_index()

        total_rows = len(temp)
        group_summary["group_percent"] = round(
            (group_summary["group_size"] / total_rows) * 100,
            2
        )

        group_summary["target_rate"] = round(
            group_summary["target_rate"] * 100,
            2
        )

        max_rate = group_summary["target_rate"].max()
        min_rate = group_summary["target_rate"].min()
        disparity = round(max_rate - min_rate, 2)

        for _, row in group_summary.iterrows():
            if disparity >= 15:
                severity = "High"
            elif disparity >= 8:
                severity = "Medium"
            else:
                severity = "Low"

            if row["group_percent"] < 5:
                representation_risk = "Underrepresented Group"
            else:
                representation_risk = "No Major Representation Issue"

            fairness_rows.append({
                "group_column": column,
                "group_value": row[column],
                "group_size": int(row["group_size"]),
                "group_percent": row["group_percent"],
                "target_rate_percent": row["target_rate"],
                "target_rate_disparity_percent": disparity,
                "fairness_severity": severity,
                "representation_risk": representation_risk
            })

    return pd.DataFrame(fairness_rows)


def generate_leakage_bias_report(leakage_df, fairness_df, output_path):
    output = []

    output.append("# Leakage & Bias Audit Report")
    output.append("")

    output.append("## Leakage Risk Summary")
    output.append("")

    if leakage_df.empty:
        output.append("- No leakage risks detected.")
    else:
        high_count = len(leakage_df[leakage_df["severity"] == "High"])
        medium_count = len(leakage_df[leakage_df["severity"] == "Medium"])

        output.append(f"- High severity leakage warnings: {high_count}")
        output.append(f"- Medium severity leakage warnings: {medium_count}")
        output.append("")

        for _, row in leakage_df.iterrows():
            output.append(
                f"- `{row['column_name']}` | {row['risk_type']} | Severity: {row['severity']}"
            )
            output.append(f"  - Reason: {row['reason']}")

    output.append("")
    output.append("## Bias / Fairness Summary")
    output.append("")

    if fairness_df.empty:
        output.append("- No group-based fairness columns detected.")
    else:
        group_columns = fairness_df["group_column"].unique().tolist()
        output.append(f"- Group columns checked: {group_columns}")
        output.append("")

        high_fairness = fairness_df[
            fairness_df["fairness_severity"] == "High"
        ]

        medium_fairness = fairness_df[
            fairness_df["fairness_severity"] == "Medium"
        ]

        output.append(f"- High fairness disparity rows: {len(high_fairness)}")
        output.append(f"- Medium fairness disparity rows: {len(medium_fairness)}")
        output.append("")

        output.append("## Full Fairness Table")
        output.append("")
        output.append(fairness_df.to_markdown(index=False))

    output.append("")
    output.append("## Required Action")
    output.append("")
    output.append("1. Remove identifier columns before modeling.")
    output.append("2. Manually review suspicious status/post-event columns.")
    output.append("3. Check whether group columns such as gender or region are ethically and legally appropriate for modeling.")
    output.append("4. Investigate groups with unusually high or low target rates.")
    output.append("5. Do not treat this as a final fairness approval. This is an initial diagnostic audit.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    input_file = "data/raw/sample_customer_ai_audit_dataset.csv"
    target_column = "churn"

    df = pd.read_csv(input_file)

    profile_df, dataset_summary = profile_dataset(
        input_file,
        target_column=target_column
    )

    leakage_df = detect_leakage_risks(
        df=df,
        profile_df=profile_df,
        target_column=target_column
    )

    fairness_df = detect_bias_and_group_risk(
        df=df,
        target_column=target_column
    )

    Path("data/exports").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    leakage_df.to_csv(
        "data/exports/leakage_warnings.csv",
        index=False
    )

    fairness_df.to_csv(
        "data/exports/bias_fairness_summary.csv",
        index=False
    )

    generate_leakage_bias_report(
        leakage_df=leakage_df,
        fairness_df=fairness_df,
        output_path="reports/leakage_bias_report.md"
    )

    print("Leakage and bias audit completed.")
    print(f"Leakage warnings: {len(leakage_df)}")
    print(f"Fairness rows checked: {len(fairness_df)}")
    print("Report saved: reports/leakage_bias_report.md")
    print("Leakage warnings saved: data/exports/leakage_warnings.csv")
    print("Fairness summary saved: data/exports/bias_fairness_summary.csv")