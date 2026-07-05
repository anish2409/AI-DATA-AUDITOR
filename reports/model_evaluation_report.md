# Production Model Evaluation Report

## Dataset Split

- Total rows: 1015
- Training rows used after safety guard: 1015
- Large dataset sampling guard applied: `False`
- Train rows: 761
- Test rows: 254
- Target column: `churn`
- Target distribution: `{0: 0.8, 1: 0.2}`

## Feature Governance

- Excluded risky columns: `['customer_id', 'last_payment_status']`
- Ignored unsupported features: `[]`
- Numeric features: `['age', 'income', 'tenure_months', 'monthly_spend', 'support_tickets']`
- Categorical features: `['gender', 'region']`

## Model Leaderboard

| model_name             |   train_f1 |   test_accuracy |   test_balanced_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   test_average_precision |   cv_f1_mean |   cv_f1_std |   cv_roc_auc_mean |   cv_roc_auc_std |   best_threshold |   best_threshold_f1 |   best_threshold_precision |   best_threshold_recall |   train_test_f1_gap | overfitting_warning   |   true_negatives |   false_positives |   false_negatives |   true_positives |
|:-----------------------|-----------:|----------------:|-------------------------:|-----------------:|--------------:|----------:|---------------:|-------------------------:|-------------:|------------:|------------------:|-----------------:|-----------------:|--------------------:|---------------------------:|------------------------:|--------------------:|:----------------------|-----------------:|------------------:|------------------:|-----------------:|
| Dummy Baseline         |     0      |          0.7992 |                   0.5    |           0      |        0      |    0      |         0.5    |                   0.2008 |       0      |      0      |            0.5    |           0      |             0.1  |              0      |                     0      |                  0      |              0      | Low Overfitting Risk  |              203 |                 0 |                51 |                0 |
| Logistic Regression    |     0.3386 |          0.5669 |                   0.5455 |           0.2342 |        0.5098 |    0.321  |         0.5826 |                   0.3033 |       0.3032 |      0.0318 |            0.5135 |           0.0254 |             0.45 |              0.339  |                     0.2162 |                  0.7843 |              0.0176 | Low Overfitting Risk  |              118 |                85 |                25 |               26 |
| Random Forest          |     0.9383 |          0.685  |                   0.5093 |           0.2157 |        0.2157 |    0.2157 |         0.5234 |                   0.2274 |       0.2528 |      0.0391 |            0.5463 |           0.0243 |             0.3  |              0.3481 |                     0.2146 |                  0.9216 |              0.7226 | High Overfitting Risk |              163 |                40 |                40 |               11 |
| Extra Trees            |     0.6618 |          0.6181 |                   0.5335 |           0.2326 |        0.3922 |    0.292  |         0.5307 |                   0.295  |       0.227  |      0.0505 |            0.5246 |           0.0357 |             0.25 |              0.346  |                     0.2101 |                  0.9804 |              0.3698 | High Overfitting Risk |              137 |                66 |                31 |               20 |
| Hist Gradient Boosting |     1      |          0.7638 |                   0.4999 |           0.2    |        0.0588 |    0.0909 |         0.53   |                   0.2268 |       0.1408 |      0.0658 |            0.5313 |           0.0368 |             0.1  |              0.2822 |                     0.2054 |                  0.451  |              0.9091 | High Overfitting Risk |              191 |                12 |                48 |                3 |

## Best Model

- Best model: **Logistic Regression**
- Test F1-score: **0.321**
- Test ROC-AUC: **0.5826**
- Test average precision: **0.3033**
- Best threshold: **0.45**
- Best threshold F1-score: **0.339**
- Overfitting warning: **Low Overfitting Risk**

## Threshold Analysis

| model_name             |   best_threshold |   best_threshold_f1 |   best_threshold_precision |   best_threshold_recall |
|:-----------------------|-----------------:|--------------------:|---------------------------:|------------------------:|
| Dummy Baseline         |             0.1  |              0      |                     0      |                  0      |
| Logistic Regression    |             0.45 |              0.339  |                     0.2162 |                  0.7843 |
| Random Forest          |             0.3  |              0.3481 |                     0.2146 |                  0.9216 |
| Extra Trees            |             0.25 |              0.346  |                     0.2101 |                  0.9804 |
| Hist Gradient Boosting |             0.1  |              0.2822 |                     0.2054 |                  0.451  |

## Top Feature Importance

| model_name          | feature_name                |   importance |
|:--------------------|:----------------------------|-------------:|
| Logistic Regression | numeric__income             |     0.377446 |
| Logistic Regression | categorical__gender_Unknown |     0.376218 |
| Logistic Regression | categorical__gender_F       |     0.323855 |
| Logistic Regression | categorical__region_North   |     0.247214 |
| Logistic Regression | categorical__region_East    |     0.197269 |
| Logistic Regression | categorical__region_north   |     0.180656 |
| Logistic Regression | categorical__gender_male    |     0.154199 |
| Logistic Regression | categorical__region_UNKNOWN |     0.120718 |
| Logistic Regression | categorical__region_South   |     0.115973 |
| Logistic Regression | categorical__region_West    |     0.110905 |
| Logistic Regression | numeric__support_tickets    |     0.087219 |
| Logistic Regression | categorical__gender_Female  |     0.059503 |
| Logistic Regression | categorical__gender_Male    |     0.047259 |
| Logistic Regression | numeric__tenure_months      |     0.028355 |
| Logistic Regression | numeric__age                |     0.019651 |
| Logistic Regression | numeric__monthly_spend      |     0.004342 |

## Production Interpretation

- Dummy Baseline is included to prove whether ML models actually beat a naive baseline.
- Cross-validation metrics reduce the risk of trusting a lucky train-test split.
- Threshold tuning is included because default 0.50 classification threshold is often not optimal.
- Feature importance helps explain which signals drive predictions.
- This evaluator is stronger than a basic student ML script, but it is still not final MLOps.
- Production deployment would still require monitoring, retraining strategy, model registry, tests, and business approval.

## Required Next Action

1. Compare model performance against Dummy Baseline.
2. Investigate false negatives if recall is weak.
3. Review top features for leakage or proxy bias.
4. Use threshold tuning based on business cost, not only F1-score.
5. Re-run evaluation after data cleaning and feature governance.