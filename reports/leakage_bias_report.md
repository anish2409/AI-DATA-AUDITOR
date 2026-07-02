# Leakage & Bias Audit Report

## Leakage Risk Summary

- High severity leakage warnings: 1
- Medium severity leakage warnings: 1

- `customer_id` | Identifier Leakage Risk | Severity: High
  - Reason: Identifier-like columns can cause memorization and should not be used directly as model features.
- `last_payment_status` | Suspicious Column Name | Severity: Medium
  - Reason: Column name contains 'last', which may indicate post-event or outcome-related information.

## Bias / Fairness Summary

- Group columns checked: ['gender', 'region']

- High fairness disparity rows: 0
- Medium fairness disparity rows: 11

## Full Fairness Table

| group_column   | group_value   |   group_size |   group_percent |   target_rate_percent |   target_rate_disparity_percent | fairness_severity   | representation_risk           |
|:---------------|:--------------|-------------:|----------------:|----------------------:|--------------------------------:|:--------------------|:------------------------------|
| gender         | F             |           68 |            7.01 |                 30.88 |                           12.34 | Medium              | No Major Representation Issue |
| gender         | Female        |          356 |           36.7  |                 18.82 |                           12.34 | Medium              | No Major Representation Issue |
| gender         | Male          |          383 |           39.48 |                 18.54 |                           12.34 | Medium              | No Major Representation Issue |
| gender         | Unknown       |           78 |            8.04 |                 19.23 |                           12.34 | Medium              | No Major Representation Issue |
| gender         | male          |           85 |            8.76 |                 24.71 |                           12.34 | Medium              | No Major Representation Issue |
| region         | East          |          171 |           16.85 |                 15.79 |                            9.05 | Medium              | No Major Representation Issue |
| region         | North         |          153 |           15.07 |                 24.84 |                            9.05 | Medium              | No Major Representation Issue |
| region         | South         |          163 |           16.06 |                 20.86 |                            9.05 | Medium              | No Major Representation Issue |
| region         | UNKNOWN       |          172 |           16.95 |                 23.26 |                            9.05 | Medium              | No Major Representation Issue |
| region         | West          |          186 |           18.33 |                 17.74 |                            9.05 | Medium              | No Major Representation Issue |
| region         | north         |          170 |           16.75 |                 18.24 |                            9.05 | Medium              | No Major Representation Issue |

## Required Action

1. Remove identifier columns before modeling.
2. Manually review suspicious status/post-event columns.
3. Check whether group columns such as gender or region are ethically and legally appropriate for modeling.
4. Investigate groups with unusually high or low target rates.
5. Do not treat this as a final fairness approval. This is an initial diagnostic audit.