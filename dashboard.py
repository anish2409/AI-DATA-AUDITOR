import json
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from src.core.full_audit_runner import run_full_audit


st.set_page_config(
    page_title="AI Data Auditor",
    page_icon="🧠",
    layout="wide"
)


RUNS_DIR = Path("audit_outputs/streamlit_runs")
UPLOADS_DIR = Path("audit_outputs/streamlit_uploads")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file, destination_dir):
    destination_dir.mkdir(parents=True, exist_ok=True)

    safe_name = uploaded_file.name.replace(" ", "_")
    output_path = destination_dir / safe_name

    with open(output_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return output_path


def read_csv_if_exists(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_text_if_exists(path):
    path = Path(path)

    if not path.exists():
        return "File not found."

    return path.read_text(encoding="utf-8")


def read_json_if_exists(path):
    path = Path(path)

    if not path.exists():
        return {}

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def make_audit_zip(run_dir):
    run_dir = Path(run_dir)
    zip_path = run_dir / "audit_package.zip"

    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, "w", ZIP_DEFLATED) as zip_file:
        for file_path in run_dir.rglob("*"):
            if file_path.is_file() and file_path.name != "audit_package.zip":
                zip_file.write(
                    file_path,
                    arcname=file_path.relative_to(run_dir)
                )

    return zip_path


def get_latest_run_dir():
    if not RUNS_DIR.exists():
        return None

    run_dirs = [
        path for path in RUNS_DIR.iterdir()
        if path.is_dir()
    ]

    if not run_dirs:
        return None

    return sorted(
        run_dirs,
        key=lambda path: path.stat().st_mtime,
        reverse=True
    )[0]


def get_run_paths(run_dir):
    run_dir = Path(run_dir)

    return {
        "manifest": run_dir / "audit_manifest.json",

        "scorecard": run_dir / "exports/final_audit_scorecard.csv",
        "ai_readiness": run_dir / "exports/ai_readiness_scores.csv",
        "drift": run_dir / "exports/drift_summary.csv",
        "model": run_dir / "exports/model_evaluation_results.csv",
        "llm": run_dir / "exports/llm_response_quality_scores.csv",

        "final_report": run_dir / "reports/final_ai_audit_summary.md",
        "data_quality_report": run_dir / "reports/data_quality_report.md",
        "ai_readiness_report": run_dir / "reports/ai_readiness_report.md",
        "drift_report": run_dir / "reports/drift_report.md",
        "leakage_bias_report": run_dir / "reports/leakage_bias_report.md",
        "model_report": run_dir / "reports/model_evaluation_report.md",
        "llm_report": run_dir / "reports/llm_response_quality_report.md",
    }


def render_header():
    st.title("AI Data Quality & Model Evaluation Auditor")

    st.write(
        "Upload a dataset, select the target column, run a full AI audit, "
        "and review data quality, AI readiness, drift, leakage, fairness, "
        "baseline model reliability, and optional LLM response quality."
    )


def render_upload_panel():
    st.sidebar.header("Run New Audit")

    current_file = st.sidebar.file_uploader(
        "Upload current/input CSV",
        type=["csv"]
    )

    preview_df = pd.DataFrame()
    target_column = None

    if current_file is not None:
        try:
            preview_df = pd.read_csv(current_file)
            current_file.seek(0)

            if preview_df.empty:
                st.sidebar.error("Uploaded CSV is empty.")
            else:
                target_column = st.sidebar.selectbox(
                    "Select target column",
                    preview_df.columns.tolist()
                )

                st.sidebar.caption(
                    f"Rows: {preview_df.shape[0]} | Columns: {preview_df.shape[1]}"
                )

        except Exception as error:
            st.sidebar.error(f"Could not read current CSV: {error}")

    baseline_file = st.sidebar.file_uploader(
        "Optional baseline CSV for drift detection",
        type=["csv"]
    )

    llm_eval_file = st.sidebar.file_uploader(
        "Optional LLM evaluation CSV",
        type=["csv"]
    )

    run_button = st.sidebar.button(
        "Run Full AI Audit",
        type="primary",
        use_container_width=True
    )

    if run_button:
        if current_file is None:
            st.sidebar.error("Upload a current/input CSV first.")
            return preview_df

        if target_column is None:
            st.sidebar.error("Select a target column.")
            return preview_df

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        upload_dir = UPLOADS_DIR / f"upload_{timestamp}"
        output_dir = RUNS_DIR / f"run_{timestamp}"

        try:
            current_file.seek(0)

            current_path = save_uploaded_file(
                uploaded_file=current_file,
                destination_dir=upload_dir
            )

            baseline_path = None

            if baseline_file is not None:
                baseline_file.seek(0)

                baseline_path = save_uploaded_file(
                    uploaded_file=baseline_file,
                    destination_dir=upload_dir
                )

            llm_path = None

            if llm_eval_file is not None:
                llm_eval_file.seek(0)

                llm_path = save_uploaded_file(
                    uploaded_file=llm_eval_file,
                    destination_dir=upload_dir
                )

            with st.spinner("Running full AI audit..."):
                result = run_full_audit(
                    input_csv_path=str(current_path),
                    target_column=target_column,
                    output_dir=str(output_dir),
                    baseline_csv_path=str(baseline_path) if baseline_path else None,
                    llm_eval_file=str(llm_path) if llm_path else None,
                )

            st.session_state["current_run_dir"] = result["output_dir"]
            st.session_state["final_decision"] = result["final_decision"]

            st.sidebar.success("Audit completed.")

        except Exception as error:
            st.sidebar.error("Audit failed.")
            st.sidebar.exception(error)

    st.sidebar.divider()

    if st.sidebar.button("Load Latest Audit Run", use_container_width=True):
        latest_run = get_latest_run_dir()

        if latest_run is None:
            st.sidebar.warning("No previous audit run found.")
        else:
            st.session_state["current_run_dir"] = str(latest_run)
            st.sidebar.success(f"Loaded: {latest_run.name}")

    return preview_df
def render_dataset_preview(preview_df):
    if preview_df.empty:
        st.info("Upload a CSV from the sidebar to preview the dataset and run an audit.")
        return

    with st.expander("Uploaded Dataset Preview", expanded=False):
        st.dataframe(preview_df.head(50), use_container_width=True)


def render_executive_summary(run_dir):
    paths = get_run_paths(run_dir)

    manifest = read_json_if_exists(paths["manifest"])
    scorecard_df = read_csv_if_exists(paths["scorecard"])
    ai_df = read_csv_if_exists(paths["ai_readiness"])
    drift_df = read_csv_if_exists(paths["drift"])
    model_df = read_csv_if_exists(paths["model"])
    llm_df = read_csv_if_exists(paths["llm"])

    st.subheader("Executive Summary")

    decision = manifest.get("final_decision", "Final decision unavailable.")

    if "Not production-ready" in decision:
        st.error(decision)
    elif "Conditionally" in decision:
        st.warning(decision)
    else:
        st.success(decision)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if not ai_df.empty and "ai_readiness_score" in ai_df.columns:
            st.metric(
                "AI Readiness Score",
                f"{ai_df.iloc[0]['ai_readiness_score']}/100"
            )
        else:
            st.metric("AI Readiness Score", "N/A")

    with col2:
        if not drift_df.empty and "drift_severity" in drift_df.columns:
            significant = len(
                drift_df[drift_df["drift_severity"] == "Significant Drift"]
            )

            st.metric("Significant Drift Columns", significant)
        else:
            st.metric("Significant Drift Columns", "Skipped")

    with col3:
        if not model_df.empty and "test_f1" in model_df.columns:
            best_model = model_df.sort_values(
                by="test_f1",
                ascending=False
            ).iloc[0]

            st.metric("Best Model F1", best_model["test_f1"])
        else:
            st.metric("Best Model F1", "N/A")

    with col4:
        if not llm_df.empty and "quality_score" in llm_df.columns:
            avg_score = round(llm_df["quality_score"].mean(), 2)
            st.metric("LLM Quality Score", f"{avg_score}/100")
        else:
            st.metric("LLM Quality Score", "Skipped")

    st.divider()

    if not scorecard_df.empty:
        st.subheader("Final Audit Scorecard")
        st.dataframe(scorecard_df, use_container_width=True)
    else:
        st.warning("Final audit scorecard not found.")


def render_drift_tab(run_dir):
    paths = get_run_paths(run_dir)
    drift_df = read_csv_if_exists(paths["drift"])

    st.subheader("Data Drift Detection")

    if drift_df.empty:
        st.warning("Drift detection was skipped or drift output was not found.")
        return

    st.dataframe(drift_df, use_container_width=True)

    if "psi" in drift_df.columns:
        psi_df = drift_df[["column_name", "psi"]].set_index("column_name")
        st.bar_chart(psi_df)


def render_model_tab(run_dir):
    paths = get_run_paths(run_dir)
    model_df = read_csv_if_exists(paths["model"])

    st.subheader("Baseline Model Evaluation")

    if model_df.empty:
        st.warning("Model evaluation results not found.")
        return

    st.dataframe(model_df, use_container_width=True)

    chart_columns = [
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "test_roc_auc"
    ]

    available_columns = [
        column for column in chart_columns
        if column in model_df.columns
    ]

    if available_columns:
        chart_df = model_df.set_index("model_name")[available_columns]
        st.bar_chart(chart_df)


def render_llm_tab(run_dir):
    paths = get_run_paths(run_dir)
    llm_df = read_csv_if_exists(paths["llm"])

    st.subheader("LLM Response Quality Evaluation")

    if llm_df.empty:
        st.warning("LLM evaluation was skipped or scores were not found.")
        return

    st.dataframe(llm_df, use_container_width=True)

    if "quality_score" in llm_df.columns:
        score_df = llm_df[["case_id", "quality_score"]].set_index("case_id")
        st.bar_chart(score_df)


def render_reports_tab(run_dir):
    paths = get_run_paths(run_dir)

    report_options = {
        "Final Audit Summary": paths["final_report"],
        "Data Quality Report": paths["data_quality_report"],
        "AI Readiness Report": paths["ai_readiness_report"],
        "Drift Report": paths["drift_report"],
        "Leakage & Bias Report": paths["leakage_bias_report"],
        "Model Evaluation Report": paths["model_report"],
        "LLM Response Quality Report": paths["llm_report"],
    }

    selected_report = st.selectbox(
        "Choose report",
        list(report_options.keys())
    )

    report_text = read_text_if_exists(report_options[selected_report])
    st.markdown(report_text)


def render_exports_tab(run_dir):
    exports_dir = Path(run_dir) / "exports"

    st.subheader("Generated CSV Exports")

    if not exports_dir.exists():
        st.warning("Exports folder not found.")
        return

    export_files = sorted(exports_dir.glob("*.csv"))

    if not export_files:
        st.warning("No CSV exports found.")
        return

    selected_file = st.selectbox(
        "Choose export file",
        export_files,
        format_func=lambda path: path.name
    )

    df = read_csv_if_exists(selected_file)

    if df.empty:
        st.warning("Selected export file is empty or unreadable.")
    else:
        st.dataframe(df, use_container_width=True)


def render_downloads(run_dir):
    st.subheader("Download Audit Package")

    run_dir = Path(run_dir)
    zip_path = make_audit_zip(run_dir)

    with open(zip_path, "rb") as file:
        st.download_button(
            label="Download Full Audit Package",
            data=file,
            file_name="ai_audit_package.zip",
            mime="application/zip",
            use_container_width=True
        )

    final_report_path = run_dir / "reports/final_ai_audit_summary.md"

    if final_report_path.exists():
        with open(final_report_path, "rb") as file:
            st.download_button(
                label="Download Final Report",
                data=file,
                file_name="final_ai_audit_summary.md",
                mime="text/markdown",
                use_container_width=True
            )
def render_run_dashboard(run_dir):
    st.caption(f"Current run: `{run_dir}`")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "Executive Summary",
            "Drift",
            "Model Evaluation",
            "LLM Evaluation",
            "Reports",
            "Exports & Downloads"
        ]
    )

    with tab1:
        render_executive_summary(run_dir)

    with tab2:
        render_drift_tab(run_dir)

    with tab3:
        render_model_tab(run_dir)

    with tab4:
        render_llm_tab(run_dir)

    with tab5:
        render_reports_tab(run_dir)

    with tab6:
        render_exports_tab(run_dir)
        st.divider()
        render_downloads(run_dir)


def main():
    render_header()

    preview_df = render_upload_panel()

    render_dataset_preview(preview_df)

    run_dir = st.session_state.get("current_run_dir")

    if run_dir is None:
        latest_run = get_latest_run_dir()

        if latest_run is not None:
            st.info(
                "No active run selected. Use the sidebar button "
                "'Load Latest Audit Run' to view the most recent audit."
            )
        else:
            st.info(
                "No audit run found yet. Upload a CSV from the sidebar "
                "and click 'Run Full AI Audit'."
            )

        return

    render_run_dashboard(run_dir)


if __name__ == "__main__":
    main()