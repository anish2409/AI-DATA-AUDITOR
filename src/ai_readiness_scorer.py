import pandas as pd
from pathlib import Path
from data_profiler import profile_dataset


def clamp_score(value):
    return round(max(0, min(100, value)), 2)


def risk_level_from_score(score):
    if score >= 85:
        return "Low Risk"
    if score >= 70:
        return "Medium Risk"
    if score >= 50:
        return "High Risk"
    return "Critical Risk"


def deployment_decision(score):
    if score >= 85:
        return "Ready for controlled AI experimentation"
    if score >= 70:
        return "Conditionally ready after cleaning and validation"
    if score >= 50:
        return "Not production-ready. Major cleaning required before modeling"
    return "Not AI-ready. Do not train or deploy models before remediation"


def calculate_ai_readiness(profile_df, dataset_summary):
    total_rows = dataset_summary["total_rows"]

    data_quality_score = dataset_summary["data_quality_score"]

    avg_missing = profile_df["missing_percent"].mean()
    max_missing = profile_df["missing_percent"].max()

    missing_safety_score = clamp_score(
        100 - min(100, avg_missing * 8 + max_missing * 4)
    )

    invalid_total = profile_df["invalid_negative_count"].sum()
    invalid_rate = (invalid_total / total_rows) * 100

    invalid_value_safety_score = clamp_score(
        100 - min(100, invalid_rate * 12)
    )

    duplicate_percent = dataset_summary["duplicate_percent"]

    duplicate_safety_score = clamp_score(
        100 - min(100, duplicate_percent * 10)
    )

    avg_outlier = profile_df["outlier_percent"].mean()
    max_outlier = profile_df["outlier_percent"].max()

    outlier_safety_score = clamp_score(
        100 - min(100, avg_outlier * 8 + max_outlier * 6)
    )

    identifier_columns = profile_df[
        profile_df["likely_identifier_flag"] == True
    ]["column_name"].tolist()

    identifier_safety_score = clamp_score(
        100 - min(40, len(identifier_columns) * 15)
    )

    imbalance_risk = dataset_summary.get("imbalance_risk", "Unknown")

    if imbalance_risk == "Low":
        imbalance_safety_score = 95
    elif imbalance_risk == "Medium":
        imbalance_safety_score = 75
    elif imbalance_risk == "High":
        imbalance_safety_score = 45
    else:
        imbalance_safety_score = 60

    ai_readiness_score = round(
        data_quality_score * 0.35
        + missing_safety_score * 0.15
        + invalid_value_safety_score * 0.15
        + duplicate_safety_score * 0.10
        + outlier_safety_score * 0.10
        + imbalance_safety_score * 0.10
        + identifier_safety_score * 0.05,
        2
    )

    scores = {
        "data_quality_score": data_quality_score,
        "missing_safety_score": missing_safety_score,
        "invalid_value_safety_score": invalid_value_safety_score,
        "duplicate_safety_score": duplicate_safety_score,
        "outlier_safety_score": outlier_safety_score,
        "imbalance_safety_score": imbalance_safety_score,
        "identifier_safety_score": identifier_safety_score,
        "ai_readiness_score": ai_readiness_score,
        "risk_level": risk_level_from_score(ai_readiness_score),
        "deployment_decision": deployment_decision(ai_readiness_score)
    }

    risks = []

    if max_missing >= 5:
        risks.append({
            "risk_type": "Missing Values",
            "severity": "Medium",
            "details": f"Maximum column missing percentage is {max_missing}%"
        })

    if invalid_total > 0:
        risks.append({
            "risk_type": "Invalid Values",
            "severity": "High",
            "details": f"{invalid_total} invalid negative values detected"
        })

    if duplicate_percent > 1:
        risks.append({
            "risk_type": "Duplicate Rows",
            "severity": "Medium",
            "details": f"{duplicate_percent}% duplicate rows detected"
        })

    if max_outlier >= 2:
        risks.append({
            "risk_type": "Outliers",
            "severity": "Medium",
            "details": f"Maximum column outlier percentage is {max_outlier}%"
        })

    if imbalance_risk in ["Medium", "High"]:
        risks.append({
            "risk_type": "Class Imbalance",
            "severity": imbalance_risk,
            "details": f"Target imbalance risk is {imbalance_risk}"
        })

    if identifier_columns:
        risks.append({
            "risk_type": "Identifier Columns",
            "severity": "Medium",
            "details": f"Identifier columns should be excluded from modeling: {identifier_columns}"
        })

    return scores, risks


def generate_ai_readiness_report(scores, risks, output_path):
    output = []

    output.append("# AI Readiness Audit Report")
    output.append("")
    output.append("## Overall AI Readiness")
    output.append("")
    output.append(f"- AI Readiness Score: **{scores['ai_readiness_score']}/100**")
    output.append(f"- Risk Level: **{scores['risk_level']}**")
    output.append(f"- Deployment Decision: **{scores['deployment_decision']}**")

    output.append("")
    output.append("## Score Breakdown")
    output.append("")
    output.append(f"- Data Quality Score: {scores['data_quality_score']}/100")
    output.append(f"- Missing Safety Score: {scores['missing_safety_score']}/100")
    output.append(f"- Invalid Value Safety Score: {scores['invalid_value_safety_score']}/100")
    output.append(f"- Duplicate Safety Score: {scores['duplicate_safety_score']}/100")
    output.append(f"- Outlier Safety Score: {scores['outlier_safety_score']}/100")
    output.append(f"- Imbalance Safety Score: {scores['imbalance_safety_score']}/100")
    output.append(f"- Identifier Safety Score: {scores['identifier_safety_score']}/100")

    output.append("")
    output.append("## Key Risks")
    output.append("")

    if not risks:
        output.append("- No major AI readiness risks detected.")
    else:
        for risk in risks:
            output.append(
                f"- **{risk['risk_type']}** | Severity: {risk['severity']} | {risk['details']}"
            )

    output.append("")
    output.append("## Required Remediation Before Modeling")
    output.append("")
    output.append("1. Remove identifier columns such as customer_id from model features.")
    output.append("2. Fix invalid negative values in age and monthly_spend.")
    output.append("3. Handle missing values in income and gender.")
    output.append("4. Investigate outliers in income and monthly_spend.")
    output.append("5. Check class imbalance before training any predictive model.")
    output.append("6. Re-run the audit after cleaning to compare before-vs-after readiness.")

    output.append("")
    output.append("## Final Verdict")
    output.append("")

    if scores["ai_readiness_score"] >= 85:
        output.append("The dataset is suitable for controlled AI experimentation.")
    elif scores["ai_readiness_score"] >= 70:
        output.append("The dataset can be used only after targeted cleaning and validation.")
    elif scores["ai_readiness_score"] >= 50:
        output.append("The dataset is not production-ready. Major remediation is required.")
    else:
        output.append("The dataset is unsafe for AI/ML usage in its current state.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    input_file = "data/raw/sample_customer_ai_audit_dataset.csv"
    target_column = "churn"

    profile_df, dataset_summary = profile_dataset(
        input_file,
        target_column=target_column
    )

    scores, risks = calculate_ai_readiness(profile_df, dataset_summary)

    Path("data/exports").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    pd.DataFrame([scores]).to_csv(
        "data/exports/ai_readiness_scores.csv",
        index=False
    )

    pd.DataFrame(risks).to_csv(
        "data/exports/ai_readiness_risks.csv",
        index=False
    )

    generate_ai_readiness_report(
        scores,
        risks,
        "reports/ai_readiness_report.md"
    )

    print("AI readiness scoring completed.")
    print(f"AI Readiness Score: {scores['ai_readiness_score']}/100")
    print(f"Risk Level: {scores['risk_level']}")
    print(f"Deployment Decision: {scores['deployment_decision']}")
    print("Report saved: reports/ai_readiness_report.md")
    print("Scores saved: data/exports/ai_readiness_scores.csv")
    print("Risks saved: data/exports/ai_readiness_risks.csv")