import pandas as pd
import pytest

from src.core.full_audit_runner import run_full_audit


def create_customer_dataset(path, drift=False):
    rows = []

    for i in range(120):
        churn = 1 if i % 5 == 0 else 0

        income = 45000 + (i * 120)
        monthly_spend = 1200 + (i * 8)
        support_tickets = i % 4

        if drift:
            income = income * 0.85
            monthly_spend = monthly_spend * 1.30
            support_tickets = support_tickets + 2

        rows.append({
            "customer_id": i + 1,
            "age": 22 + (i % 45),
            "income": income,
            "gender": "Male" if i % 2 == 0 else "Female",
            "region": ["North", "South", "East", "West"][i % 4],
            "tenure_months": 1 + (i % 60),
            "monthly_spend": monthly_spend,
            "support_tickets": support_tickets,
            "last_payment_status": "paid" if i % 7 != 0 else "failed",
            "churn": churn,
        })

    df = pd.DataFrame(rows)

    df.loc[0, "income"] = None
    df.loc[1, "age"] = -5
    df.loc[2, "monthly_spend"] = -100

    df.to_csv(path, index=False)


def create_llm_eval_dataset(path):
    df = pd.DataFrame([
        {
            "case_id": 1,
            "prompt": "Can customer_id be used as a model feature?",
            "reference_answer": "Identifier columns should be excluded because they can cause leakage.",
            "model_answer": "customer_id should be excluded because it can cause leakage. Source: Feature Safety Guide.",
            "reference_context": "Feature Safety Guide: identifier columns should be excluded from model features.",
            "expected_keywords": "identifier,excluded,leakage",
            "required_citation": "Feature Safety Guide",
            "latency_ms": 900,
            "estimated_cost_usd": 0.002,
        },
        {
            "case_id": 2,
            "prompt": "What does high PSI mean?",
            "reference_answer": "High PSI indicates significant distribution drift.",
            "model_answer": "High PSI means the model is definitely production ready.",
            "reference_context": "Drift Guide: PSI above 0.25 indicates significant drift.",
            "expected_keywords": "psi,drift,distribution",
            "required_citation": "Drift Guide",
            "latency_ms": 1100,
            "estimated_cost_usd": 0.003,
        },
    ])

    df.to_csv(path, index=False)


def test_full_audit_runner_creates_required_artifacts(tmp_path):
    baseline_path = tmp_path / "baseline.csv"
    current_path = tmp_path / "current.csv"
    llm_path = tmp_path / "llm_eval.csv"
    output_dir = tmp_path / "audit_run"

    create_customer_dataset(baseline_path, drift=False)
    create_customer_dataset(current_path, drift=True)
    create_llm_eval_dataset(llm_path)

    result = run_full_audit(
        input_csv_path=str(current_path),
        target_column="churn",
        output_dir=str(output_dir),
        baseline_csv_path=str(baseline_path),
        llm_eval_file=str(llm_path),
    )

    assert output_dir.exists()
    assert result["final_decision"]

    assert (output_dir / "audit_manifest.json").exists()

    assert (output_dir / "reports" / "final_ai_audit_summary.md").exists()
    assert (output_dir / "reports" / "data_quality_report.md").exists()
    assert (output_dir / "reports" / "model_evaluation_report.md").exists()

    assert (output_dir / "exports" / "final_audit_scorecard.csv").exists()
    assert (output_dir / "exports" / "column_quality_summary.csv").exists()
    assert (output_dir / "exports" / "ai_readiness_scores.csv").exists()
    assert (output_dir / "exports" / "drift_summary.csv").exists()
    assert (output_dir / "exports" / "model_evaluation_results.csv").exists()
    assert (output_dir / "exports" / "model_threshold_analysis.csv").exists()
    assert (output_dir / "exports" / "model_feature_importance.csv").exists()
    assert (output_dir / "exports" / "llm_response_quality_scores.csv").exists()

    scorecard = pd.read_csv(output_dir / "exports" / "final_audit_scorecard.csv")

    assert not scorecard.empty
    assert "module" in scorecard.columns
    assert "key_finding" in scorecard.columns


def test_full_audit_runner_rejects_missing_target(tmp_path):
    current_path = tmp_path / "current.csv"
    output_dir = tmp_path / "audit_run"

    create_customer_dataset(current_path, drift=False)

    with pytest.raises(ValueError):
        run_full_audit(
            input_csv_path=str(current_path),
            target_column="wrong_target",
            output_dir=str(output_dir),
        )