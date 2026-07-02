# Baseline Model Evaluation Report

## Dataset Split

- Total rows: 1015
- Training rows: 761
- Testing rows: 254
- Target column: `churn`
- Excluded risky columns: `['customer_id', 'last_payment_status']`

## Feature Set

- Numeric features: `['age', 'income', 'tenure_months', 'monthly_spend', 'support_tickets']`
- Categorical features: `['gender', 'region']`

## Model Comparison

| model_name          |   train_f1 |   test_accuracy |   test_precision |   test_recall |   test_f1 |   test_roc_auc |   train_test_f1_gap | overfitting_warning       |   true_negatives |   false_positives |   false_negatives |   true_positives |
|:--------------------|-----------:|----------------:|-----------------:|--------------:|----------:|---------------:|--------------------:|:--------------------------|-----------------:|------------------:|------------------:|-----------------:|
| Logistic Regression |     0.3386 |          0.5669 |           0.2342 |        0.5098 |    0.321  |         0.5826 |              0.0176 | Low Overfitting Risk      |              118 |                85 |                25 |               26 |
| Decision Tree       |     0.4297 |          0.3976 |           0.1928 |        0.6275 |    0.2949 |         0.4563 |              0.1347 | Moderate Overfitting Risk |               69 |               134 |                19 |               32 |
| Random Forest       |     0.9408 |          0.6772 |           0.1556 |        0.1373 |    0.1458 |         0.5013 |              0.795  | High Overfitting Risk     |              165 |                38 |                44 |                7 |

## Best Baseline Model

- Best model by test F1-score: **Logistic Regression**
- Test F1-score: **0.321**
- Test ROC-AUC: **0.5826**
- Overfitting warning: **Low Overfitting Risk**

## Interpretation

- This is a baseline model evaluation, not a final production model.
- Identifier columns and suspicious post-event columns were excluded before training.
- Missing values were imputed using median or most frequent strategy.
- Categorical columns were one-hot encoded.
- If recall is low, the model may miss positive churn cases.
- If train-test F1 gap is high, the model may be overfitting.
- Final model training should happen only after data cleaning, leakage review, drift monitoring, and fairness checks.

## Required Next Action

1. Clean invalid negative values in age and monthly_spend.
2. Normalize inconsistent categories such as Male/male and F/Female.
3. Re-run data quality and AI readiness audits after cleaning.
4. Compare model performance before and after cleaning.
5. Continue with LLM response evaluation module after baseline model evaluation.