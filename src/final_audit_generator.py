import pandas as pd
from pathlib import Path


REPORT_PATH = "reports/final_ai_audit_summary.md"
SCORECARD_PATH = "data/exports/final_audit_scorecard.csv"


def safe_read_csv(path):
    if Path(path).exists():
        return pd.read_csv(path)

    return pd.DataFrame()


def get_ai_readiness_summary():
    scores_df = safe_read_csv("data/exports/ai_readiness_scores.csv")

    if scores_df.empty:
        return {
            "score": None,
            "risk": "Unknown",
            "decision": "AI readiness score not available"
        }

    row = scores_df.iloc[0]

    return {
        "score": row.get("ai_readiness_score"),
        "risk": row.get("risk_level"),
        "decision": row.get("deployment_decision")
    }


def get_drift_summary():
    drift_df = safe_read_csv("data/exports/drift_summary.csv")

    if drift_df.empty:
        return {
            "moderate": 0,
            "significant": 0,
            "finding": "Drift report not available"
        }

    moderate = len(drift_df[drift_df["drift_severity"] == "Moderate Drift"])
    significant = len(drift_df[drift_df["drift_severity"] == "Significant Drift"])

    finding = f"{moderate} moderate drift columns and {significant} significant drift columns detected"

    return {
        "moderate": moderate,
        "significant": significant,
        "finding": finding
    }


def get_leakage_bias_summary():
    leakage_df = safe_read_csv("data/exports/leakage_warnings.csv")
    fairness_df = safe_read_csv("data/exports/bias_fairness_summary.csv")

    leakage_count = len(leakage_df)
    high_leakage_count = 0

    if not leakage_df.empty and "severity" in leakage_df.columns:
        high_leakage_count = len(leakage_df[leakage_df["severity"] == "High"])

    fairness_rows = len(fairness_df)

    finding = f"{leakage_count} leakage warnings, {high_leakage_count} high severity, {fairness_rows} fairness rows checked"

    return {
        "leakage_count": leakage_count,
        "high_leakage_count": high_leakage_count,
        "fairness_rows": fairness_rows,
        "finding": finding
    }


def get_model_summary():
    model_df = safe_read_csv("data/exports/model_evaluation_results.csv")

    if model_df.empty:
        return {
            "best_model": "Unknown",
            "best_f1": None,
            "best_auc": None,
            "finding": "Model evaluation not available"
        }

    best_row = model_df.sort_values(by="test_f1", ascending=False).iloc[0]

    finding = (
        f"Best baseline model is {best_row['model_name']} "
        f"with F1-score {best_row['test_f1']} and ROC-AUC {best_row['test_roc_auc']}"
    )

    return {
        "best_model": best_row["model_name"],
        "best_f1": best_row["test_f1"],
        "best_auc": best_row["test_roc_auc"],
        "finding": finding
    }


def get_llm_summary():
    llm_df = safe_read_csv("data/exports/llm_response_quality_scores.csv")

    if llm_df.empty:
        return {
            "avg_score": None,
            "failed": 0,
            "finding": "LLM response evaluation not available"
        }

    avg_score = round(llm_df["quality_score"].mean(), 2)

    failed = len(
        llm_df[
            llm_df["final_decision"].str.contains("Fail", na=False)
        ]
    )

    finding = f"Average LLM response quality score is {avg_score}/100 with {failed} failed responses"

    return {
        "avg_score": avg_score,
        "failed": failed,
        "finding": finding
    }


def final_decision(ai_summary, drift_summary, leakage_summary, model_summary, llm_summary):
    risk_points = 0

    if ai_summary["score"] is not None and ai_summary["score"] < 70:
        risk_points += 2

    if drift_summary["significant"] > 0:
        risk_points += 2

    if leakage_summary["high_leakage_count"] > 0:
        risk_points += 2

    if model_summary["best_f1"] is not None and model_summary["best_f1"] < 0.5:
        risk_points += 1

    if llm_summary["failed"] > 0:
        risk_points += 1

    if risk_points >= 6:
        return "Not production-ready. Major remediation required before AI deployment."

    if risk_points >= 3:
        return "Conditionally usable for experimentation only. Not ready for production."

    return "Low-risk experimental dataset. Still requires human review before production."


def generate_scorecard(ai_summary, drift_summary, leakage_summary, model_summary, llm_summary):
    rows = [
        {
            "module": "Data Quality & AI Readiness",
            "status": ai_summary["risk"],
            "score": ai_summary["score"],
            "key_finding": ai_summary["decision"]
        },
        {
            "module": "Data Drift Detection",
            "status": "Risk Detected" if drift_summary["significant"] > 0 else "Stable",
            "score": "-",
            "key_finding": drift_summary["finding"]
        },
        {
            "module": "Leakage & Bias Audit",
            "status": "Risk Detected" if leakage_summary["leakage_count"] > 0 else "No Major Risk",
            "score": "-",
            "key_finding": leakage_summary["finding"]
        },
        {
            "module": "Baseline Model Evaluation",
            "status": "Weak Baseline" if model_summary["best_f1"] and model_summary["best_f1"] < 0.5 else "Acceptable Baseline",
            "score": model_summary["best_f1"],
            "key_finding": model_summary["finding"]
        },
        {
            "module": "LLM Response Evaluation",
            "status": "Risk Detected" if llm_summary["failed"] > 0 else "Passed",
            "score": llm_summary["avg_score"],
            "key_finding": llm_summary["finding"]
        }
    ]

    return pd.DataFrame(rows)


def generate_final_report(scorecard_df, decision, output_path):
    output = []

    output.append("# Final AI Audit Summary Report")
    output.append("")

    output.append("## Executive Decision")
    output.append("")
    output.append(f"**{decision}**")

    output.append("")
    output.append("## Audit Scorecard")
    output.append("")
    output.append(scorecard_df.to_markdown(index=False))

    output.append("")
    output.append("## Key Business Interpretation")
    output.append("")
    output.append("- The dataset is useful for experimentation, but it is not safe for production AI deployment yet.")
    output.append("- Data quality issues, drift, leakage risks, fairness disparity, and weak model performance were detected.")
    output.append("- The LLM response evaluator also found failed or risky responses.")
    output.append("- This proves why AI systems need audit checks before deployment.")

    output.append("")
    output.append("## Recommended Next Actions")
    output.append("")
    output.append("1. Clean invalid negative values.")
    output.append("2. Normalize inconsistent categories.")
    output.append("3. Remove identifier and suspicious leakage columns.")
    output.append("4. Re-run AI readiness scoring after cleaning.")
    output.append("5. Re-train baseline models after remediation.")
    output.append("6. Review failed LLM responses manually.")
    output.append("7. Build a Streamlit dashboard for demo presentation.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    ai_summary = get_ai_readiness_summary()
    drift_summary = get_drift_summary()
    leakage_summary = get_leakage_bias_summary()
    model_summary = get_model_summary()
    llm_summary = get_llm_summary()

    decision = final_decision(
        ai_summary=ai_summary,
        drift_summary=drift_summary,
        leakage_summary=leakage_summary,
        model_summary=model_summary,
        llm_summary=llm_summary
    )

    scorecard_df = generate_scorecard(
        ai_summary=ai_summary,
        drift_summary=drift_summary,
        leakage_summary=leakage_summary,
        model_summary=model_summary,
        llm_summary=llm_summary
    )

    Path("data/exports").mkdir(parents=True, exist_ok=True)

    scorecard_df.to_csv(SCORECARD_PATH, index=False)

    generate_final_report(
        scorecard_df=scorecard_df,
        decision=decision,
        output_path=REPORT_PATH
    )

    print("Final AI audit summary generated.")
    print(f"Decision: {decision}")
    print(f"Report saved: {REPORT_PATH}")
    print(f"Scorecard saved: {SCORECARD_PATH}")