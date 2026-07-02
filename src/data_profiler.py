import pandas as pd
from pathlib import Path


def detect_column_type(series):
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical_or_text"


def calculate_outlier_count(series):
    if not pd.api.types.is_numeric_dtype(series):
        return 0

    clean_series = series.dropna()

    if clean_series.empty:
        return 0

    q1 = clean_series.quantile(0.25)
    q3 = clean_series.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return 0

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return int(((clean_series < lower_bound) | (clean_series > upper_bound)).sum())


def calculate_data_quality_score(profile_df, dataset_summary):
    avg_missing = profile_df["missing_percent"].mean()
    max_missing = profile_df["missing_percent"].max()
    duplicate_percent = dataset_summary["duplicate_percent"]
    invalid_total = profile_df["invalid_negative_count"].sum()
    total_rows = dataset_summary["total_rows"]
    avg_outlier = profile_df["outlier_percent"].mean()
    max_outlier = profile_df["outlier_percent"].max()

    missing_penalty = min(30, avg_missing * 2.5 + max_missing * 0.8)
    duplicate_penalty = min(15, duplicate_percent * 3)
    invalid_penalty = min(25, (invalid_total / total_rows) * 100 * 8)
    outlier_penalty = min(15, avg_outlier * 3 + max_outlier * 1.5)

    medium_missing_columns = len(profile_df[profile_df["missing_percent"] >= 5])
    issue_count_penalty = min(15, medium_missing_columns * 3)

    score = 100 - (
        missing_penalty
        + duplicate_penalty
        + invalid_penalty
        + outlier_penalty
        + issue_count_penalty
    )

    return round(max(0, min(100, score)), 2)


def profile_dataset(input_path, target_column=None):
    df = pd.read_csv(input_path)

    total_rows = len(df)
    total_columns = len(df.columns)
    duplicate_rows = int(df.duplicated().sum())

    column_profiles = []

    for column in df.columns:
        series = df[column]

        missing_count = int(series.isna().sum())
        missing_percent = round((missing_count / total_rows) * 100, 2)

        unique_count = int(series.nunique(dropna=True))
        unique_percent = round((unique_count / total_rows) * 100, 2)

        column_type = detect_column_type(series)

        outlier_count = calculate_outlier_count(series)
        outlier_percent = round((outlier_count / total_rows) * 100, 2)

        invalid_negative_count = 0

        if pd.api.types.is_numeric_dtype(series):
            invalid_negative_count = int((series.dropna() < 0).sum())

        constant_flag = unique_count <= 1

        likely_identifier_flag = (
            column_type == "numeric"
            and unique_percent > 90
            and (column.lower() == "id" or column.lower().endswith("_id") or "uuid" in column.lower() or column.lower().endswith("_key"))
        )

        high_cardinality_flag = (
            unique_percent > 80
            and column_type == "categorical_or_text"
        )

        column_profiles.append({
            "column_name": column,
            "detected_type": column_type,
            "missing_count": missing_count,
            "missing_percent": missing_percent,
            "unique_count": unique_count,
            "unique_percent": unique_percent,
            "outlier_count": outlier_count,
            "outlier_percent": outlier_percent,
            "invalid_negative_count": invalid_negative_count,
            "constant_flag": constant_flag,
            "likely_identifier_flag": likely_identifier_flag,
            "high_cardinality_flag": high_cardinality_flag
        })

    profile_df = pd.DataFrame(column_profiles)

    dataset_summary = {
        "input_path": input_path,
        "total_rows": total_rows,
        "total_columns": total_columns,
        "duplicate_rows": duplicate_rows,
        "duplicate_percent": round((duplicate_rows / total_rows) * 100, 2),
        "target_column": target_column
    }

    if target_column and target_column in df.columns:
        target_distribution = df[target_column].value_counts(normalize=True).round(4).to_dict()
        dataset_summary["target_distribution"] = target_distribution

        minority_class_percent = min(target_distribution.values()) * 100
        dataset_summary["minority_class_percent"] = round(minority_class_percent, 2)

        if minority_class_percent < 10:
            dataset_summary["imbalance_risk"] = "High"
        elif minority_class_percent < 25:
            dataset_summary["imbalance_risk"] = "Medium"
        else:
            dataset_summary["imbalance_risk"] = "Low"
    else:
        dataset_summary["target_distribution"] = None
        dataset_summary["minority_class_percent"] = None
        dataset_summary["imbalance_risk"] = "Unknown"

    dataset_summary["data_quality_score"] = calculate_data_quality_score(
        profile_df,
        dataset_summary
    )

    return profile_df, dataset_summary


def generate_markdown_report(profile_df, dataset_summary, output_path):
    output = []

    output.append("# Data Quality Audit Report")
    output.append("")
    output.append("## Dataset Summary")
    output.append("")
    output.append(f"- Input file: `{dataset_summary['input_path']}`")
    output.append(f"- Total rows: {dataset_summary['total_rows']}")
    output.append(f"- Total columns: {dataset_summary['total_columns']}")
    output.append(f"- Duplicate rows: {dataset_summary['duplicate_rows']} ({dataset_summary['duplicate_percent']}%)")
    output.append(f"- Data Quality Score: **{dataset_summary['data_quality_score']}/100**")

    if dataset_summary.get("target_column"):
        output.append(f"- Target column: `{dataset_summary['target_column']}`")
        output.append(f"- Target distribution: `{dataset_summary['target_distribution']}`")
        output.append(f"- Imbalance risk: **{dataset_summary['imbalance_risk']}**")

    output.append("")
    output.append("## Major Issues")
    output.append("")

    missing_issues = profile_df[profile_df["missing_percent"] >= 3]
    invalid_negative = profile_df[profile_df["invalid_negative_count"] > 0]
    outlier_heavy = profile_df[profile_df["outlier_percent"] >= 1]
    likely_identifiers = profile_df[profile_df["likely_identifier_flag"] == True]
    high_cardinality = profile_df[profile_df["high_cardinality_flag"] == True]

    if (
        missing_issues.empty
        and invalid_negative.empty
        and outlier_heavy.empty
        and likely_identifiers.empty
        and high_cardinality.empty
    ):
        output.append("- No major issues detected.")
    else:
        for _, row in missing_issues.iterrows():
            output.append(f"- `{row['column_name']}` has missing values: {row['missing_percent']}%")

        for _, row in invalid_negative.iterrows():
            output.append(f"- `{row['column_name']}` has invalid negative values: {row['invalid_negative_count']} records")

        for _, row in outlier_heavy.iterrows():
            output.append(f"- `{row['column_name']}` has possible outliers: {row['outlier_percent']}%")

        for _, row in likely_identifiers.iterrows():
            output.append(f"- `{row['column_name']}` looks like an identifier and should not be used directly as a model feature")

        for _, row in high_cardinality.iterrows():
            output.append(f"- `{row['column_name']}` has high cardinality: {row['unique_percent']}% unique values")

    output.append("")
    output.append("## Column Quality Summary")
    output.append("")
    output.append(profile_df.to_markdown(index=False))

    output.append("")
    output.append("## Initial Recommendation")
    output.append("")

    score = dataset_summary["data_quality_score"]

    if score >= 85:
        output.append("- Dataset quality is strong, but final AI readiness still requires leakage, drift, imbalance, and model evaluation checks.")
    elif score >= 70:
        output.append("- Dataset is usable but needs cleaning before model training or production AI usage.")
    elif score >= 50:
        output.append("- Dataset has significant quality issues. Cleaning and validation are mandatory before modeling.")
    else:
        output.append("- Dataset is not AI-ready. Do not train models before fixing major quality issues.")

    output.append("")
    output.append("## Scoring Note")
    output.append("")
    output.append("- The data quality score penalizes missing values, duplicate rows, invalid negative values, outliers, and repeated column-level issues.")
    output.append("- This score is only the first audit layer. AI readiness will be calculated separately in the next phase.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    input_file = "data/raw/sample_customer_ai_audit_dataset.csv"
    target_column = "churn"

    profile_df, summary = profile_dataset(input_file, target_column)

    Path("data/exports").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    profile_df.to_csv("data/exports/column_quality_summary.csv", index=False)

    generate_markdown_report(
        profile_df,
        summary,
        "reports/data_quality_report.md"
    )

    print("Data profiling completed.")
    print(f"Data Quality Score: {summary['data_quality_score']}/100")
    print("Report saved: reports/data_quality_report.md")
    print("Column summary saved: data/exports/column_quality_summary.csv")
