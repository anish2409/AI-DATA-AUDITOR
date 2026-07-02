# Data Drift Detection Report

## Drift Summary

- Total columns checked: 10
- Moderate drift columns: 2
- Significant drift columns: 2

## Drifted Columns

- `support_tickets` | Significant Drift | PSI: 0.4097
- `monthly_spend` | Significant Drift | PSI: 0.3183
- `income` | Moderate Drift | PSI: 0.1902
- `region` | Moderate Drift | PSI: 0.1588

## Missing Value Drift

- `income` missing changed from 8.08% to 12.81%

## Full Drift Table

| column_name         | column_role   | data_type           |   baseline_missing_percent |   current_missing_percent |   missing_drift_percent |   baseline_mean |   current_mean | baseline_top_value   | current_top_value   |    psi |   ks_statistic |   ks_pvalue | drift_severity       |
|:--------------------|:--------------|:--------------------|---------------------------:|--------------------------:|------------------------:|----------------:|---------------:|:---------------------|:--------------------|-------:|---------------:|------------:|:---------------------|
| customer_id         | feature       | numeric             |                       0    |                      0    |                    0    |          502.3  |         502.3  | nan                  | nan                 | 0      |         0      |     1       | No Significant Drift |
| age                 | feature       | numeric             |                       0    |                      0    |                    0    |           34.8  |          34.8  | nan                  | nan                 | 0      |         0      |     1       | No Significant Drift |
| income              | feature       | numeric             |                       8.08 |                     12.81 |                    4.73 |        53664.3  |       47155.1  | nan                  | nan                 | 0.1902 |         0.1784 |     0       | Moderate Drift       |
| gender              | feature       | categorical_or_text |                       4.43 |                      4.43 |                    0    |          nan    |         nan    | Male                 | Male                | 0      |       nan      |   nan       | No Significant Drift |
| region              | feature       | categorical_or_text |                       0    |                      0    |                    0    |          nan    |         nan    | West                 | West                | 0.1588 |       nan      |   nan       | Moderate Drift       |
| tenure_months       | feature       | numeric             |                       0    |                      0    |                    0    |           35.49 |          35.49 | nan                  | nan                 | 0      |         0      |     1       | No Significant Drift |
| monthly_spend       | feature       | numeric             |                       2.96 |                      2.96 |                    0    |         2713.26 |        3335.78 | nan                  | nan                 | 0.3183 |         0.2315 |     0       | Significant Drift    |
| support_tickets     | feature       | numeric             |                       0    |                      0    |                    0    |            1.92 |           2.94 | nan                  | nan                 | 0.4097 |         0.2611 |     0       | Significant Drift    |
| last_payment_status | feature       | categorical_or_text |                       0    |                      0    |                    0    |          nan    |         nan    | paid                 | paid                | 0.087  |       nan      |   nan       | No Significant Drift |
| churn               | target        | numeric             |                       0    |                      0    |                    0    |            0.2  |           0.26 | nan                  | nan                 | 0.0231 |         0.064  |     0.03111 | No Significant Drift |

## Interpretation

- PSI < 0.10 means no significant drift.
- PSI between 0.10 and 0.25 means moderate drift.
- PSI > 0.25 means significant drift.
- Drifted features should be investigated before model retraining or deployment.