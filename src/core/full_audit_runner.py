import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))


from data_profiler import profile_dataset
from ai_readiness_scorer import calculate_ai_readiness, generate_ai_readiness_report
from drift_detector import detect_drift, generate_drift_report
from leakage_bias_checker import (
    detect_leakage_risks,
    detect_bias_and_group_risk,
    generate_leakage_bias_report,
)
import model_evaluator as model_eval
from llm_response_evaluator import evaluate_llm_responses, generate_report as generate_llm_report


SUSPICIOUS_EXCLUSION_TERMS = [
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
    "resolved",
]


def make_run_dirs(output_dir):
    output_path = Path(output_dir)

    reports_dir = output_path / "reports"
    exports_dir = output_path / "exports"
    inputs_dir = output_path / "inputs"

    reports_dir.mkdir(parents=True, exist_ok=True)
    exports_dir.mkdir(parents=True, exist_ok=True)
    inputs_dir.mkdir(parents=True, exist_ok=True)

    return output_path, reports_dir, exports_dir, inputs_dir


def validate_input_file(input_path, target_column):
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if input_file.suffix.lower() != ".csv":
        raise ValueError("Only CSV files are supported right now.")

    df = pd.read_csv(input_file)

    if df.empty:
        raise ValueError("Input CSV is empty.")

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def write_data_quality_report(profile_df, dataset_summary, output_path):
    output = []

    output.append("# Data Quality Audit Report")
    output.append("")

    output.append("## Dataset Summary")
    output.append("")
    output.append(f"- Total rows: {dataset_summary.get('total_rows')}")
    output.append(f"- Total columns: {dataset_summary.get('total_columns')}")
    output.append(f"- Duplicate rows: {dataset_summary.get('duplicate_rows')} ({dataset_summary.get('duplicate_percent')}%)")
    output.append(f"- Data Quality Score: **{dataset_summary.get('data_quality_score')}/100**")

    if dataset_summary.get("target_column"):
        output.append(f"- Target column: `{dataset_summary.get('target_column')}`")
        output.append(f"- Target distribution: `{dataset_summary.get('target_distribution')}`")
        output.append(f"- Imbalance risk: **{dataset_summary.get('imbalance_risk')}**")

    output.append("")
    output.append("## Column Quality Summary")
    output.append("")
    output.append(profile_df.to_markdown(index=False))

    output.append("")
    output.append("## Interpretation")
    output.append("")
    output.append("- Missing values, invalid values, duplicates, outliers, and identifier-like columns reduce AI readiness.")
    output.append("- This report should be reviewed before model training or AI deployment.")

    Path(output_path).write_text("\n".join(output), encoding="utf-8")


def get_identifier_columns(profile_df):
    if "likely_identifier_flag" not in profile_df.columns:
        return []

    return profile_df[
        profile_df["likely_identifier_flag"] == True
    ]["column_name"].tolist()


def get_suspicious_columns(df, target_column):
    suspicious_columns = []

    for column in df.columns:
        if column == target_column:
            continue

        column_lower = column.lower()

        for term in SUSPICIOUS_EXCLUSION_TERMS:
            if term in column_lower:
                suspicious_columns.append(column)
                break

    return suspicious_columns


def get_model_excluded_columns(df, profile_df, target_column):
    identifier_columns = get_identifier_columns(profile_df)
    suspicious_columns = get_suspicious_columns(df, target_column)

    excluded_columns = sorted(
        set(identifier_columns + suspicious_columns)
    )

    return excluded_columns


def run_data_quality(input_path, target_column, reports_dir, exports_dir):
    profile_df, dataset_summary = profile_dataset(
        input_path,
        target_column=target_column
    )

    profile_output = exports_dir / "column_quality_summary.csv"
    report_output = reports_dir / "data_quality_report.md"

    profile_df.to_csv(profile_output, index=False)

    write_data_quality_report(
        profile_df=profile_df,
        dataset_summary=dataset_summary,
        output_path=report_output
    )

    return profile_df, dataset_summary


def run_ai_readiness(profile_df, dataset_summary, reports_dir, exports_dir):
    scores, risks = calculate_ai_readiness(
        profile_df=profile_df,
        dataset_summary=dataset_summary
    )

    scores_df = pd.DataFrame([scores])
    risks_df = pd.DataFrame(risks)

    scores_path = exports_dir / "ai_readiness_scores.csv"
    risks_path = exports_dir / "ai_readiness_risks.csv"
    report_path = reports_dir / "ai_readiness_report.md"

    scores_df.to_csv(scores_path, index=False)
    risks_df.to_csv(risks_path, index=False)

    generate_ai_readiness_report(
        scores=scores,
        risks=risks,
        output_path=report_path
    )

    return scores, risks_df


def run_drift_if_available(baseline_path, current_path, target_column, reports_dir, exports_dir):
    if baseline_path is None:
        return {
            "enabled": False,
            "moderate": 0,
            "significant": 0,
            "message": "Drift detection skipped because baseline file was not provided."
        }

    baseline_file = Path(baseline_path)

    if not baseline_file.exists():
        return {
            "enabled": False,
            "moderate": 0,
            "significant": 0,
            "message": f"Drift detection skipped because baseline file was not found: {baseline_path}"
        }

    drift_df = detect_drift(
        baseline_path=baseline_path,
        current_path=current_path,
        target_column=target_column
    )

    drift_path = exports_dir / "drift_summary.csv"
    report_path = reports_dir / "drift_report.md"

    drift_df.to_csv(drift_path, index=False)

    generate_drift_report(
        drift_df=drift_df,
        output_path=report_path
    )

    moderate = len(drift_df[drift_df["drift_severity"] == "Moderate Drift"])
    significant = len(drift_df[drift_df["drift_severity"] == "Significant Drift"])

    return {
        "enabled": True,
        "moderate": moderate,
        "significant": significant,
        "message": f"{moderate} moderate drift columns and {significant} significant drift columns detected."
    }


def run_leakage_bias(df, profile_df, target_column, reports_dir, exports_dir):
    leakage_df = detect_leakage_risks(
        df=df,
        profile_df=profile_df,
        target_column=target_column
    )

    fairness_df = detect_bias_and_group_risk(
        df=df,
        target_column=target_column
    )

    leakage_path = exports_dir / "leakage_warnings.csv"
    fairness_path = exports_dir / "bias_fairness_summary.csv"
    report_path = reports_dir / "leakage_bias_report.md"

    leakage_df.to_csv(leakage_path, index=False)
    fairness_df.to_csv(fairness_path, index=False)

    generate_leakage_bias_report(
        leakage_df=leakage_df,
        fairness_df=fairness_df,
        output_path=report_path
    )

    high_leakage_count = 0

    if not leakage_df.empty and "severity" in leakage_df.columns:
        high_leakage_count = len(leakage_df[leakage_df["severity"] == "High"])

    return {
        "leakage_count": len(leakage_df),
        "high_leakage_count": high_leakage_count,
        "fairness_rows": len(fairness_df),
    }


def run_model_evaluation(df, target_column, excluded_columns, reports_dir, exports_dir):
    model_eval.TARGET_COLUMN = target_column
    model_eval.EXCLUDED_COLUMNS = excluded_columns

    results_df, metadata = model_eval.evaluate_models(df)

    result_path = exports_dir / "model_evaluation_results.csv"
    report_path = reports_dir / "model_evaluation_report.md"

    results_df.to_csv(result_path, index=False)

    model_eval.generate_model_report(
        results_df=results_df,
        metadata=metadata,
        output_path=report_path
    )

    best_model = results_df.sort_values(
        by="test_f1",
        ascending=False
    ).iloc[0]

    return {
        "best_model": best_model["model_name"],
        "best_f1": best_model["test_f1"],
        "best_auc": best_model["test_roc_auc"],
        "overfitting_warning": best_model["overfitting_warning"],
        "excluded_columns": excluded_columns,
    }


def run_llm_eval_if_available(llm_eval_file, reports_dir, exports_dir):
    if llm_eval_file is None:
        return {
            "enabled": False,
            "avg_score": None,
            "failed": 0,
            "message": "LLM evaluation skipped because file was not provided."
        }

    llm_path = Path(llm_eval_file)

    if not llm_path.exists():
        return {
            "enabled": False,
            "avg_score": None,
            "failed": 0,
            "message": f"LLM evaluation skipped because file was not found: {llm_eval_file}"
        }

    llm_df = pd.read_csv(llm_path)

    results_df = evaluate_llm_responses(llm_df)

    scores_path = exports_dir / "llm_response_quality_scores.csv"
    report_path = reports_dir / "llm_response_quality_report.md"

    results_df.to_csv(scores_path, index=False)

    generate_llm_report(
        results_df=results_df,
        output_path=report_path
    )

    avg_score = round(results_df["quality_score"].mean(), 2)

    failed = len(
        results_df[
            results_df["final_decision"].str.contains("Fail", na=False)
        ]
    )

    return {
        "enabled": True,
        "avg_score": avg_score,
        "failed": failed,
        "message": f"Average LLM response quality score is {avg_score}/100 with {failed} failed responses."
    }


def build_scorecard(ai_scores, drift_summary, leakage_summary, model_summary, llm_summary):
    rows = [
        {
            "module": "Data Quality & AI Readiness",
            "status": ai_scores.get("risk_level"),
            "score": ai_scores.get("ai_readiness_score"),
            "key_finding": ai_scores.get("deployment_decision"),
        },
        {
            "module": "Data Drift Detection",
            "status": "Risk Detected" if drift_summary["significant"] > 0 else "Stable / Skipped",
            "score": "-",
            "key_finding": drift_summary["message"],
        },
        {
            "module": "Leakage & Bias Audit",
            "status": "Risk Detected" if leakage_summary["leakage_count"] > 0 else "No Major Risk",
            "score": "-",
            "key_finding": (
                f"{leakage_summary['leakage_count']} leakage warnings, "
                f"{leakage_summary['high_leakage_count']} high severity, "
                f"{leakage_summary['fairness_rows']} fairness rows checked"
            ),
        },
        {
            "module": "Baseline Model Evaluation",
            "status": "Weak Baseline" if model_summary["best_f1"] < 0.5 else "Acceptable Baseline",
            "score": model_summary["best_f1"],
            "key_finding": (
                f"Best model is {model_summary['best_model']} "
                f"with F1-score {model_summary['best_f1']} "
                f"and ROC-AUC {model_summary['best_auc']}"
            ),
        },
        {
            "module": "LLM Response Evaluation",
            "status": "Risk Detected" if llm_summary["failed"] > 0 else "Passed / Skipped",
            "score": llm_summary["avg_score"] if llm_summary["avg_score"] is not None else "-",
            "key_finding": llm_summary["message"],
        },
    ]

    return pd.DataFrame(rows)


def make_final_decision(ai_scores, drift_summary, leakage_summary, model_summary, llm_summary):
    risk_points = 0

    if ai_scores.get("ai_readiness_score", 0) < 70:
        risk_points += 2

    if drift_summary["enabled"] and drift_summary["significant"] > 0:
        risk_points += 2

    if leakage_summary["high_leakage_count"] > 0:
        risk_points += 2

    if model_summary["best_f1"] < 0.5:
        risk_points += 1

    if llm_summary["enabled"] and llm_summary["failed"] > 0:
        risk_points += 1

    if risk_points >= 6:
        return "Not production-ready. Major remediation required before AI deployment."

    if risk_points >= 3:
        return "Conditionally usable for experimentation only. Not ready for production."

    return "Low-risk experimental dataset. Human review still required before production."


def write_final_report(scorecard_df, final_decision, output_path):
    output = []

    output.append("# Final AI Audit Summary Report")
    output.append("")

    output.append("## Executive Decision")
    output.append("")
    output.append(f"**{final_decision}**")

    output.append("")
    output.append("## Audit Scorecard")
    output.append("")
    output.append(scorecard_df.to_markdown(index=False))

    output.append("")
    output.append("## Production Interpretation")
    output.append("")
    output.append("- This audit runner validates whether the uploaded dataset is safe for AI experimentation or deployment.")
    output.append("- It checks data quality, AI readiness, drift, leakage, fairness, baseline model reliability, and optional LLM response quality.")
    output.append("- A production approval should not be based only on model accuracy.")
    output.append("- High drift, leakage, fairness disparity, weak model results, or failed LLM responses require remediation.")

    output.append("")
    output.append("## Recommended Next Actions")
    output.append("")
    output.append("1. Clean invalid and missing values.")
    output.append("2. Normalize inconsistent categories.")
    output.append("3. Remove identifier and leakage-risk columns.")
    output.append("4. Re-run the full audit after remediation.")
    output.append("5. Compare before-vs-after audit scorecards.")
    output.append("6. Approve deployment only after human review and business validation.")

    Path(output_path).write_text("\n".join(output), encoding="utf-8")


def write_manifest(output_path, manifest):
    manifest_path = Path(output_path) / "audit_manifest.json"

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8"
    )


def run_full_audit(
    input_csv_path,
    target_column,
    output_dir,
    baseline_csv_path=None,
    llm_eval_file=None,
):
    output_path, reports_dir, exports_dir, inputs_dir = make_run_dirs(output_dir)

    df = validate_input_file(
        input_path=input_csv_path,
        target_column=target_column
    )

    copied_input_path = inputs_dir / Path(input_csv_path).name
    shutil.copy2(input_csv_path, copied_input_path)

    if baseline_csv_path is not None and Path(baseline_csv_path).exists():
        shutil.copy2(
            baseline_csv_path,
            inputs_dir / f"baseline_{Path(baseline_csv_path).name}"
        )

    profile_df, dataset_summary = run_data_quality(
        input_path=input_csv_path,
        target_column=target_column,
        reports_dir=reports_dir,
        exports_dir=exports_dir
    )

    ai_scores, ai_risks_df = run_ai_readiness(
        profile_df=profile_df,
        dataset_summary=dataset_summary,
        reports_dir=reports_dir,
        exports_dir=exports_dir
    )

    drift_summary = run_drift_if_available(
        baseline_path=baseline_csv_path,
        current_path=input_csv_path,
        target_column=target_column,
        reports_dir=reports_dir,
        exports_dir=exports_dir
    )

    leakage_summary = run_leakage_bias(
        df=df,
        profile_df=profile_df,
        target_column=target_column,
        reports_dir=reports_dir,
        exports_dir=exports_dir
    )

    excluded_columns = get_model_excluded_columns(
        df=df,
        profile_df=profile_df,
        target_column=target_column
    )

    model_summary = run_model_evaluation(
        df=df,
        target_column=target_column,
        excluded_columns=excluded_columns,
        reports_dir=reports_dir,
        exports_dir=exports_dir
    )

    llm_summary = run_llm_eval_if_available(
        llm_eval_file=llm_eval_file,
        reports_dir=reports_dir,
        exports_dir=exports_dir
    )

    scorecard_df = build_scorecard(
        ai_scores=ai_scores,
        drift_summary=drift_summary,
        leakage_summary=leakage_summary,
        model_summary=model_summary,
        llm_summary=llm_summary
    )

    scorecard_path = exports_dir / "final_audit_scorecard.csv"

    scorecard_df.to_csv(scorecard_path, index=False)

    decision = make_final_decision(
        ai_scores=ai_scores,
        drift_summary=drift_summary,
        leakage_summary=leakage_summary,
        model_summary=model_summary,
        llm_summary=llm_summary
    )

    final_report_path = reports_dir / "final_ai_audit_summary.md"

    write_final_report(
        scorecard_df=scorecard_df,
        final_decision=decision,
        output_path=final_report_path
    )

    manifest = {
        "run_id": output_path.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_csv": str(input_csv_path),
        "baseline_csv": str(baseline_csv_path) if baseline_csv_path else None,
        "target_column": target_column,
        "output_dir": str(output_path),
        "final_decision": decision,
        "reports_dir": str(reports_dir),
        "exports_dir": str(exports_dir),
        "model_excluded_columns": excluded_columns,
        "modules": {
            "data_quality": True,
            "ai_readiness": True,
            "drift_detection": drift_summary["enabled"],
            "leakage_bias": True,
            "model_evaluation": True,
            "llm_response_evaluation": llm_summary["enabled"],
        },
    }

    write_manifest(
        output_path=output_path,
        manifest=manifest
    )

    return {
        "output_dir": str(output_path),
        "final_decision": decision,
        "scorecard_path": str(scorecard_path),
        "final_report_path": str(final_report_path),
        "manifest": manifest,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run full AI data audit pipeline."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to current/input CSV file."
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target column name."
    )

    parser.add_argument(
        "--output",
        default="audit_outputs/latest_run",
        help="Output directory for audit reports and exports."
    )

    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional baseline CSV for drift detection."
    )

    parser.add_argument(
        "--llm-eval-file",
        default="data/llm_eval/llm_response_eval_dataset.csv",
        help="Optional LLM evaluation dataset CSV."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    result = run_full_audit(
        input_csv_path=args.input,
        target_column=args.target,
        output_dir=args.output,
        baseline_csv_path=args.baseline,
        llm_eval_file=args.llm_eval_file,
    )

    print("Full AI audit completed.")
    print(f"Decision: {result['final_decision']}")
    print(f"Output directory: {result['output_dir']}")
    print(f"Final report: {result['final_report_path']}")
    print(f"Scorecard: {result['scorecard_path']}")


if __name__ == "__main__":
    main()