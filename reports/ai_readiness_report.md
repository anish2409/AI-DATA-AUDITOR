# AI Readiness Audit Report

## Overall AI Readiness

- AI Readiness Score: **67.77/100**
- Risk Level: **High Risk**
- Deployment Decision: **Not production-ready. Major cleaning required before modeling**

## Score Breakdown

- Data Quality Score: 57.94/100
- Missing Safety Score: 55.3/100
- Invalid Value Safety Score: 70.44/100
- Duplicate Safety Score: 85.2/100
- Outlier Safety Score: 83.64/100
- Imbalance Safety Score: 75/100
- Identifier Safety Score: 85/100

## Key Risks

- **Missing Values** | Severity: Medium | Maximum column missing percentage is 8.08%
- **Invalid Values** | Severity: High | 25 invalid negative values detected
- **Duplicate Rows** | Severity: Medium | 1.48% duplicate rows detected
- **Outliers** | Severity: Medium | Maximum column outlier percentage is 2.07%
- **Class Imbalance** | Severity: Medium | Target imbalance risk is Medium
- **Identifier Columns** | Severity: Medium | Identifier columns should be excluded from modeling: ['customer_id']

## Required Remediation Before Modeling

1. Remove identifier columns such as customer_id from model features.
2. Fix invalid negative values in age and monthly_spend.
3. Handle missing values in income and gender.
4. Investigate outliers in income and monthly_spend.
5. Check class imbalance before training any predictive model.
6. Re-run the audit after cleaning to compare before-vs-after readiness.

## Final Verdict

The dataset is not production-ready. Major remediation is required.