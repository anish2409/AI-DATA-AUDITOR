# Data Quality Audit Report

## Dataset Summary

- Input file: `data/raw/sample_customer_ai_audit_dataset.csv`
- Total rows: 1015
- Total columns: 10
- Duplicate rows: 15 (1.48%)
- Data Quality Score: **57.94/100**
- Target column: `churn`
- Target distribution: `{0: 0.8, 1: 0.2}`
- Imbalance risk: **Medium**

## Major Issues

- `income` has missing values: 8.08%
- `gender` has missing values: 4.43%
- `age` has invalid negative values: 12 records
- `monthly_spend` has invalid negative values: 13 records
- `age` has possible outliers: 1.48%
- `income` has possible outliers: 1.18%
- `monthly_spend` has possible outliers: 2.07%
- `customer_id` looks like an identifier and should not be used directly as a model feature

## Column Quality Summary

| column_name         | detected_type       |   missing_count |   missing_percent |   unique_count |   unique_percent |   outlier_count |   outlier_percent |   invalid_negative_count | constant_flag   | likely_identifier_flag   | high_cardinality_flag   |
|:--------------------|:--------------------|----------------:|------------------:|---------------:|-----------------:|----------------:|------------------:|-------------------------:|:----------------|:-------------------------|:------------------------|
| customer_id         | numeric             |               0 |              0    |           1000 |            98.52 |               0 |              0    |                        0 | False           | True                     | False                   |
| age                 | numeric             |               0 |              0    |             69 |             6.8  |              15 |              1.48 |                       12 | False           | False                    | False                   |
| income              | numeric             |              82 |              8.08 |            916 |            90.25 |              12 |              1.18 |                        0 | False           | False                    | False                   |
| gender              | categorical_or_text |              45 |              4.43 |              5 |             0.49 |               0 |              0    |                        0 | False           | False                    | False                   |
| region              | categorical_or_text |               0 |              0    |              6 |             0.59 |               0 |              0    |                        0 | False           | False                    | False                   |
| tenure_months       | numeric             |               0 |              0    |             72 |             7.09 |               0 |              0    |                        0 | False           | False                    | False                   |
| monthly_spend       | numeric             |              30 |              2.96 |            958 |            94.38 |              21 |              2.07 |                       13 | False           | False                    | False                   |
| support_tickets     | numeric             |               0 |              0    |              9 |             0.89 |               2 |              0.2  |                        0 | False           | False                    | False                   |
| last_payment_status | categorical_or_text |               0 |              0    |              3 |             0.3  |               0 |              0    |                        0 | False           | False                    | False                   |
| churn               | numeric             |               0 |              0    |              2 |             0.2  |               0 |              0    |                        0 | False           | False                    | False                   |

## Initial Recommendation

- Dataset has significant quality issues. Cleaning and validation are mandatory before modeling.

## Scoring Note

- The data quality score penalizes missing values, duplicate rows, invalid negative values, outliers, and repeated column-level issues.
- This score is only the first audit layer. AI readiness will be calculated separately in the next phase.