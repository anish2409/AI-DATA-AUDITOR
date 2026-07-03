import pandas as pd
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="AI Data Auditor",
    page_icon="🧠",
    layout="wide"
)


REPORTS = {
    "Final Audit Summary": "reports/final_ai_audit_summary.md",
    "Data Quality Report": "reports/data_quality_report.md",
    "AI Readiness Report": "reports/ai_readiness_report.md",
    "Drift Report": "reports/drift_report.md",
    "Leakage & Bias Report": "reports/leakage_bias_report.md",
    "Model Evaluation Report": "reports/model_evaluation_report.md",
    "LLM Response Quality Report": "reports/llm_response_quality_report.md",
}


EXPORTS = {
    "Final Scorecard": "data/exports/final_audit_scorecard.csv",
    "AI Readiness Scores": "data/exports/ai_readiness_scores.csv",
    "Drift Summary": "data/exports/drift_summary.csv",
    "Leakage Warnings": "data/exports/leakage_warnings.csv",
    "Bias Fairness Summary": "data/exports/bias_fairness_summary.csv",
    "Model Evaluation Results": "data/exports/model_evaluation_results.csv",
    "LLM Response Scores": "data/exports/llm_response_quality_scores.csv",
}


def read_markdown(path):
    report_path = Path(path)

    if report_path.exists():
        return report_path.read_text(encoding="utf-8")

    return "Report not found. Please run the audit scripts first."


def read_csv(path):
    csv_path = Path(path)

    if csv_path.exists():
        return pd.read_csv(csv_path)

    return pd.DataFrame()


def load_scorecard():
    return read_csv(EXPORTS["Final Scorecard"])


def load_ai_readiness():
    return read_csv(EXPORTS["AI Readiness Scores"])


def load_model_results():
    return read_csv(EXPORTS["Model Evaluation Results"])


def load_llm_scores():
    return read_csv(EXPORTS["LLM Response Scores"])


def load_drift_summary():
    return read_csv(EXPORTS["Drift Summary"])


def show_header():
    st.title("AI Data Quality & Model Evaluation Auditor")

    st.write(
        "A production-style audit dashboard for checking data quality, "
        "AI readiness, drift, leakage, fairness, model performance, "
        "and LLM response quality."
    )


def show_executive_metrics():
    scorecard_df = load_scorecard()
    ai_df = load_ai_readiness()
    model_df = load_model_results()
    llm_df = load_llm_scores()
    drift_df = load_drift_summary()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not ai_df.empty:
            st.metric(
                label="AI Readiness Score",
                value=f"{ai_df.iloc[0]['ai_readiness_score']}/100"
            )
        else:
            st.metric("AI Readiness Score", "N/A")

    with col2:
        if not drift_df.empty:
            significant_drift = len(
                drift_df[drift_df["drift_severity"] == "Significant Drift"]
            )
            st.metric(
                label="Significant Drift Columns",
                value=significant_drift
            )
        else:
            st.metric("Significant Drift Columns", "N/A")

    with col3:
        if not model_df.empty:
            best_model = model_df.sort_values(
                by="test_f1",
                ascending=False
            ).iloc[0]

            st.metric(
                label="Best Model F1",
                value=best_model["test_f1"]
            )
        else:
            st.metric("Best Model F1", "N/A")

    with col4:
        if not llm_df.empty:
            avg_llm_score = round(llm_df["quality_score"].mean(), 2)

            st.metric(
                label="LLM Quality Score",
                value=f"{avg_llm_score}/100"
            )
        else:
            st.metric("LLM Quality Score", "N/A")

    st.divider()

    if not scorecard_df.empty:
        st.subheader("Final Audit Scorecard")
        st.dataframe(scorecard_df, use_container_width=True)
    else:
        st.warning("Final audit scorecard not found. Run final_audit_generator.py first.")


def show_model_section():
    model_df = load_model_results()

    st.subheader("Baseline Model Evaluation")

    if model_df.empty:
        st.warning("Model evaluation results not found.")
        return

    st.dataframe(model_df, use_container_width=True)

    chart_df = model_df.set_index("model_name")[
        ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    ]

    st.bar_chart(chart_df)


def show_drift_section():
    drift_df = load_drift_summary()

    st.subheader("Data Drift Detection")

    if drift_df.empty:
        st.warning("Drift summary not found.")
        return

    st.dataframe(drift_df, use_container_width=True)

    if "psi" in drift_df.columns:
        psi_df = drift_df[["column_name", "psi"]].set_index("column_name")
        st.bar_chart(psi_df)


def show_llm_section():
    llm_df = load_llm_scores()

    st.subheader("LLM Response Quality Evaluation")

    if llm_df.empty:
        st.warning("LLM response quality scores not found.")
        return

    st.dataframe(llm_df, use_container_width=True)

    score_df = llm_df[["case_id", "quality_score"]].set_index("case_id")
    st.bar_chart(score_df)


def show_report_viewer():
    st.subheader("Audit Reports")

    selected_report = st.selectbox(
        "Choose a report",
        list(REPORTS.keys())
    )

    report_text = read_markdown(REPORTS[selected_report])

    st.markdown(report_text)


def show_export_viewer():
    st.subheader("CSV Export Viewer")

    selected_export = st.selectbox(
        "Choose an export file",
        list(EXPORTS.keys())
    )

    df = read_csv(EXPORTS[selected_export])

    if df.empty:
        st.warning("Selected export file not found or empty.")
    else:
        st.dataframe(df, use_container_width=True)


def main():
    show_header()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Executive Summary",
            "Drift",
            "Model Evaluation",
            "LLM Evaluation",
            "Reports & Exports"
        ]
    )

    with tab1:
        show_executive_metrics()

    with tab2:
        show_drift_section()

    with tab3:
        show_model_section()

    with tab4:
        show_llm_section()

    with tab5:
        report_tab, export_tab = st.tabs(["Reports", "Exports"])

        with report_tab:
            show_report_viewer()

        with export_tab:
            show_export_viewer()


if __name__ == "__main__":
    main()