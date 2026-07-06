# AI Data Quality & Model Evaluation Auditor

A production-style AI audit platform for evaluating whether a dataset, machine learning workflow, and LLM response workflow are safe enough for AI experimentation or deployment.

This project is not just a model training notebook. It is an end-to-end audit system that checks data quality, AI readiness, drift, leakage risk, fairness signals, baseline model reliability, LLM response quality, and final deployment readiness.

---

## Why This Project Exists

Most AI projects fail because teams jump directly into model training without checking whether the data is clean, stable, fair, leakage-free, and production-safe.

This project solves that problem by answering:

- Is the uploaded dataset clean enough for AI modeling?
- Are there missing values, duplicates, invalid values, outliers, or identifier columns?
- Is the dataset AI-ready?
- Has the current data drifted from baseline data?
- Are there leakage-risk columns?
- Are there fairness or group-disparity risks?
- Do baseline ML models actually beat a dummy baseline?
- Are LLM responses grounded, cited, and safe?
- Should this system move toward production or stay in experimentation?

---

## Current Final Decision Example

For the included sample audit workflow, the system returns:

```text
Not production-ready. Major remediation required before AI deployment.
```

This is intentional. The project demonstrates that weak data quality, drift, leakage risk, fairness disparity, and unreliable model behavior should block production AI deployment.

---

## Key Features

### Dynamic Upload-Based Audit Dashboard

The Streamlit dashboard allows users to:

- Upload a current/input CSV
- Select the target column
- Optionally upload a baseline CSV for drift detection
- Optionally upload an LLM evaluation CSV
- Run the full AI audit from the browser
- View audit history
- Load previous audit runs
- Inspect audit metadata
- View reports and CSV exports
- Download the full audit package

---

### Full Audit Runner

The full audit runner executes the entire pipeline from one function or CLI command.

Core function:

```python
run_full_audit(
    input_csv_path,
    target_column,
    output_dir,
    baseline_csv_path=None,
    llm_eval_file=None
)
```

CLI example:

```bash
python -m src.core.full_audit_runner --input data/drift_samples/current_customer_ai_audit_dataset.csv --baseline data/raw/sample_customer_ai_audit_dataset.csv --target churn --output audit_outputs/sample_run
```

Each run creates:

```text
audit_outputs/
└── run_id/
    ├── audit_manifest.json
    ├── inputs/
    ├── reports/
    └── exports/
```

---

## Audit Modules

### 1. Data Quality Profiler

Checks:

- Missing values
- Duplicate rows
- Invalid negative values
- Outliers
- Identifier-like columns
- Target imbalance

Outputs:

```text
data_quality_report.md
column_quality_summary.csv
```

---

### 2. AI Readiness Scorer

Calculates an AI readiness score based on:

- Data quality score
- Missing value safety
- Invalid value safety
- Duplicate risk
- Outlier risk
- Class imbalance
- Identifier leakage risk

Outputs:

```text
ai_readiness_report.md
ai_readiness_scores.csv
ai_readiness_risks.csv
```

---

### 3. Data Drift Detector

Compares baseline and current datasets using:

- PSI
- KS test
- Missing-value drift

Outputs:

```text
drift_report.md
drift_summary.csv
```

---

### 4. Leakage & Bias Checker

Detects:

- Identifier leakage risk
- Suspicious post-event columns
- Group-wise target disparity
- Fairness-risk signals by group columns such as gender and region

Outputs:

```text
leakage_bias_report.md
leakage_warnings.csv
bias_fairness_summary.csv
```

---

### 5. Production Model Evaluator

This module is stronger than a basic student ML script.

It includes:

- Dummy baseline
- Logistic Regression
- Random Forest
- Extra Trees
- Hist Gradient Boosting
- Train/test metrics
- Cross-validation metrics
- ROC-AUC
- Average precision
- Balanced accuracy
- Threshold tuning
- Overfitting gap detection
- Feature importance export
- Large dataset sampling guard

Outputs:

```text
model_evaluation_report.md
model_evaluation_results.csv
model_threshold_analysis.csv
model_feature_importance.csv
```

Important: the goal is not to force a high score. The goal is to prove whether the model is actually reliable enough to trust.

---

### 6. LLM Response Quality Evaluator

Evaluates AI-generated responses using:

- Keyword coverage
- Citation presence
- Reference answer overlap
- Contradiction risk
- Latency risk
- Cost risk
- Final pass/fail decision

Outputs:

```text
llm_response_quality_report.md
llm_response_quality_scores.csv
```

---

### 7. Final Audit Summary

Combines all audit modules into an executive-level scorecard and final decision.

Outputs:

```text
final_ai_audit_summary.md
final_audit_scorecard.csv
audit_manifest.json
```

---

## Dashboard Workflow

1. Open the Streamlit dashboard.
2. Upload the current/input CSV.
3. Select the target column.
4. Optionally upload a baseline CSV.
5. Optionally upload an LLM evaluation CSV.
6. Click `Run Full AI Audit`.
7. Review the executive decision, audit scorecard, drift results, model leaderboard, LLM quality results, reports, and exports.
8. Download the full audit package.

Run locally:

```bash
streamlit run dashboard.py
```

---

## Project Architecture

```text
AI-DATA-AUDITOR/
│
├── dashboard.py
│
├── src/
│   ├── core/
│   │   └── full_audit_runner.py
│   │
│   ├── data_profiler.py
│   ├── ai_readiness_scorer.py
│   ├── drift_detector.py
│   ├── leakage_bias_checker.py
│   ├── model_evaluator.py
│   ├── llm_response_evaluator.py
│   ├── final_audit_generator.py
│   ├── create_sample_data.py
│   ├── create_drift_sample.py
│   └── create_llm_eval_data.py
│
├── tests/
│   └── test_full_audit_runner.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       └── docker-build.yml
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-prod.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Streamlit
- Pytest
- GitHub Actions
- Docker configuration
- Markdown reporting
- CSV audit exports

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/anish2409/AI-DATA-AUDITOR.git
cd AI-DATA-AUDITOR
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate virtual environment

Windows PowerShell:

```bash
.\venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Generate sample data

```bash
python src/create_sample_data.py
python src/create_drift_sample.py
python src/create_llm_eval_data.py
```

### 6. Run dashboard

```bash
streamlit run dashboard.py
```

---

## Run Full Audit from CLI

```bash
python -m src.core.full_audit_runner --input data/drift_samples/current_customer_ai_audit_dataset.csv --baseline data/raw/sample_customer_ai_audit_dataset.csv --target churn --output audit_outputs/sample_run
```

---

## Run Tests

```bash
python -m pytest -q
```

Expected result:

```text
2 passed
```

---

## CI/CD

This repository includes GitHub Actions workflows for automated validation.

### Python Test CI

```text
.github/workflows/ci.yml
```

Runs automated tests on push and pull request.

### Docker Build Check

```text
.github/workflows/docker-build.yml
```

Verifies that the project can build as a Docker image in GitHub Actions.

---

## Docker Support

This repository includes Docker deployment configuration.

Files:

```text
Dockerfile
docker-compose.yml
requirements-prod.txt
.dockerignore
```

Build command:

```bash
docker build -t ai-data-auditor .
```

Run command:

```bash
docker run --rm -p 8501:8501 ai-data-auditor
```

Docker build is verified through GitHub Actions.

---

## Example Audit Outputs

The audit system generates:

| Output | Purpose |
|---|---|
| `data_quality_report.md` | Data quality issues |
| `ai_readiness_report.md` | AI readiness score |
| `drift_report.md` | Baseline vs current data drift |
| `leakage_bias_report.md` | Leakage and fairness risk |
| `model_evaluation_report.md` | Model leaderboard and reliability analysis |
| `llm_response_quality_report.md` | LLM response quality scoring |
| `final_ai_audit_summary.md` | Final executive decision |
| `audit_manifest.json` | Run metadata and enabled modules |

---

## Business Value

This project demonstrates how organizations can audit AI workflows before deployment.

It is relevant for:

- Data Analyst roles
- Data Scientist roles
- AI/ML Developer roles
- AI Governance projects
- Responsible AI systems
- Model Risk Management
- LLM evaluation workflows
- Data quality automation
- MLOps readiness checks

---

## What Makes This Different From a Basic ML Project

A normal student ML project usually does this:

```text
Load data → train model → show accuracy
```

This project does more:

```text
Upload data
→ validate data quality
→ score AI readiness
→ detect drift
→ detect leakage and fairness risks
→ train multiple baseline models
→ compare against dummy baseline
→ tune thresholds
→ export feature importance
→ evaluate LLM responses
→ generate final audit decision
→ provide downloadable reports
→ run automated tests
→ pass CI and Docker build checks
```

---

## Current Limitations

This is a serious portfolio-grade project, but it is not a full enterprise SaaS platform yet.

Current limitations:

- No authentication
- No database-backed user accounts
- No model registry
- No scheduled monitoring jobs
- No background task queue
- No production cloud deployment yet
- Fairness checks are diagnostic, not legal compliance approval
- LLM evaluator is rule-based, not a replacement for human review

These limitations are intentional and clearly documented.

---

## Future Improvements

- PDF export
- SHAP-based model explainability
- Config-driven audit rules
- Database-backed audit history
- User authentication
- Cloud deployment
- Background audit jobs
- Larger dataset benchmarking
- Advanced fairness metrics
- Model registry integration
- LLM-as-judge evaluation mode

---

## Resume Bullet

Built a production-style AI Data Quality & Model Evaluation Auditor using Python, Streamlit, Pandas, Scikit-learn, Pytest, GitHub Actions, and Docker configuration to run upload-based AI audits, detect data quality issues, AI readiness risks, drift, leakage, fairness disparity, weak model reliability, and LLM response failures, generating downloadable reports, model leaderboards, feature importance, threshold analysis, and final deployment-readiness decisions.

---

## Status

Current status:

```text
Production-style portfolio project
Tests passing
Docker build verified through GitHub Actions
Merged into main branch
Ready for final screenshot and portfolio presentation
```