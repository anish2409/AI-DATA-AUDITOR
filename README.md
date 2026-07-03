# AI Data Quality & Model Evaluation Auditor

A production-style AI audit system that evaluates whether a dataset, machine learning model, and LLM response workflow are safe enough for AI experimentation or deployment.

This project focuses on AI-ready data validation, drift detection, leakage risk, bias checks, baseline model evaluation, LLM response quality scoring, and final audit reporting.

---

## Problem Statement

Most AI and ML projects jump directly into model training without checking whether the data is clean, safe, unbiased, stable, and production-ready.

This project solves that problem by building an end-to-end AI audit pipeline that answers:

- Is the dataset clean enough for AI modeling?
- Are there missing values, duplicates, invalid values, and outliers?
- Is the dataset AI-ready?
- Has the data distribution drifted?
- Are there leakage or bias risks?
- Are baseline ML models reliable?
- Are LLM responses grounded, cited, and safe?
- Is the system ready for production or only experimentation?

---

## Project Modules

### 1. Data Quality Profiler

Analyzes raw data and detects:

- Missing values
- Duplicate rows
- Invalid negative values
- Outliers
- Identifier columns
- Target imbalance

Output:

- `reports/data_quality_report.md`
- `data/exports/column_quality_summary.csv`

---

### 2. AI Readiness Scorer

Creates an AI readiness score based on:

- Data quality
- Missing value safety
- Invalid value safety
- Duplicate risk
- Outlier risk
- Class imbalance
- Identifier leakage risk

Output:

- `reports/ai_readiness_report.md`
- `data/exports/ai_readiness_scores.csv`
- `data/exports/ai_readiness_risks.csv`

Current result:

- AI Readiness Score: `67.77/100`
- Risk Level: `High Risk`
- Decision: `Not production-ready`

---

### 3. Data Drift Detector

Compares baseline data with current data and detects distribution shift using PSI and KS test.

Detected drift:

- `support_tickets` — Significant Drift
- `monthly_spend` — Significant Drift
- `income` — Moderate Drift
- `region` — Moderate Drift

Output:

- `reports/drift_report.md`
- `data/exports/drift_summary.csv`

---

### 4. Leakage & Bias Checker

Detects:

- Identifier leakage
- Suspicious post-event columns
- Group-wise target disparity
- Fairness risk by gender and region

Output:

- `reports/leakage_bias_report.md`
- `data/exports/leakage_warnings.csv`
- `data/exports/bias_fairness_summary.csv`

Detected risks:

- `customer_id` — High leakage risk
- `last_payment_status` — Suspicious post-event feature
- Medium fairness disparity across `gender` and `region`

---

### 5. Baseline Model Evaluator

Trains and evaluates baseline ML models after excluding risky columns.

Models used:

- Logistic Regression
- Decision Tree
- Random Forest

Best baseline model:

- Model: `Logistic Regression`
- F1-score: `0.321`
- ROC-AUC: `0.5826`

Output:

- `reports/model_evaluation_report.md`
- `data/exports/model_evaluation_results.csv`

Interpretation:

The weak baseline performance proves that dirty and risky datasets should not be trusted for production AI deployment without remediation.

---

### 6. LLM Response Quality Evaluator

Evaluates AI/LLM responses using rule-based checks:

- Keyword coverage
- Citation presence
- Reference answer overlap
- Contradiction risk
- Latency risk
- Cost risk
- Final pass/fail decision

Output:

- `reports/llm_response_quality_report.md`
- `data/exports/llm_response_quality_scores.csv`

Current result:

- Average LLM Quality Score: `68.49/100`
- Passed Responses: `5`
- Failed Responses: `3`

---

### 7. Final AI Audit Summary

Combines all module outputs into one executive-level audit report.

Output:

- `reports/final_ai_audit_summary.md`
- `data/exports/final_audit_scorecard.csv`

Final decision:

**Not production-ready. Major remediation required before AI deployment.**

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Markdown reporting
- Git & GitHub
- PowerShell
- VS Code

---

## Project Structure

```text
AI-DATA-AUDITOR/
│
├── data/
│   └── exports/
│
├── reports/
│
├── src/
│   ├── create_sample_data.py
│   ├── data_profiler.py
│   ├── ai_readiness_scorer.py
│   ├── create_drift_sample.py
│   ├── drift_detector.py
│   ├── leakage_bias_checker.py
│   ├── model_evaluator.py
│   ├── create_llm_eval_data.py
│   ├── llm_response_evaluator.py
│   └── final_audit_generator.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## How to Run

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

For Windows PowerShell:

```bash
.\venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Generate sample data

```bash
python src/create_sample_data.py
```

### 5. Run data quality profiler

```bash
python src/data_profiler.py
```

### 6. Run AI readiness scorer

```bash
python src/ai_readiness_scorer.py
```

### 7. Generate drift sample

```bash
python src/create_drift_sample.py
```

### 8. Run drift detector

```bash
python src/drift_detector.py
```

### 9. Run leakage and bias checker

```bash
python src/leakage_bias_checker.py
```

### 10. Run baseline model evaluator

```bash
python src/model_evaluator.py
```

### 11. Generate LLM evaluation data

```bash
python src/create_llm_eval_data.py
```

### 12. Run LLM response evaluator

```bash
python src/llm_response_evaluator.py
```

### 13. Generate final audit summary

```bash
python src/final_audit_generator.py
```

---

## Key Reports

| Report | Purpose |
|---|---|
| `data_quality_report.md` | Data quality issues |
| `ai_readiness_report.md` | AI readiness score |
| `drift_report.md` | Drift monitoring |
| `leakage_bias_report.md` | Leakage and fairness risks |
| `model_evaluation_report.md` | Baseline ML model results |
| `llm_response_quality_report.md` | LLM answer quality |
| `final_ai_audit_summary.md` | Final executive audit decision |

---

## Business Value

This project demonstrates how companies can audit AI systems before deployment.

It can be useful for:

- Data Analyst roles
- AI/ML Developer roles
- Data Science roles
- AI Governance projects
- Model Risk Management workflows
- LLM evaluation workflows
- Responsible AI dashboards

---

## Resume Bullet

Built an end-to-end AI Data Quality & Model Evaluation Auditor using Python, Pandas, Scikit-learn, and SciPy to detect data quality issues, AI readiness risks, drift, leakage, fairness disparity, weak baseline model performance, and LLM response quality failures, generating executive audit reports and deployment readiness decisions.

---

## Future Improvements

- Streamlit dashboard
- Automated PDF report export
- Real dataset support
- Config-driven audit rules
- Model explainability with SHAP
- Advanced fairness metrics
- LLM-as-judge integration
- CI pipeline for automated audit checks

---

## Final Status

This project is currently in active development.

Completed:

- Data Quality Profiler
- AI Readiness Scorer
- Drift Detector
- Leakage & Bias Checker
- Baseline Model Evaluator
- LLM Response Quality Evaluator
- Final AI Audit Summary Generator

Next:

- Streamlit dashboard
- Screenshots
- Deployment