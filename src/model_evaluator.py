import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


TARGET_COLUMN = "churn"

EXCLUDED_COLUMNS = [
    "customer_id",
    "last_payment_status"
]


def load_dataset(input_path):
    df = pd.read_csv(input_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    return df


def prepare_features(df):
    excluded_present = [
        column for column in EXCLUDED_COLUMNS
        if column in df.columns
    ]

    feature_df = df.drop(columns=[TARGET_COLUMN] + excluded_present)
    target = df[TARGET_COLUMN]

    numeric_features = feature_df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = feature_df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    return feature_df, target, numeric_features, categorical_features, excluded_present


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features)
        ]
    )

    return preprocessor


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def evaluate_single_model(model_name, model, preprocessor, x_train, x_test, y_train, y_test):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(x_train, y_train)

    train_predictions = pipeline.predict(x_train)
    test_predictions = pipeline.predict(x_test)

    if hasattr(pipeline.named_steps["model"], "predict_proba"):
        test_probabilities = pipeline.predict_proba(x_test)[:, 1]
        roc_auc = roc_auc_score(y_test, test_probabilities)
    else:
        roc_auc = np.nan

    train_f1 = f1_score(y_train, train_predictions, zero_division=0)
    test_f1 = f1_score(y_test, test_predictions, zero_division=0)

    train_test_gap = train_f1 - test_f1

    if train_test_gap >= 0.20:
        overfitting_warning = "High Overfitting Risk"
    elif train_test_gap >= 0.10:
        overfitting_warning = "Moderate Overfitting Risk"
    else:
        overfitting_warning = "Low Overfitting Risk"

    cm = confusion_matrix(y_test, test_predictions)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    result = {
        "model_name": model_name,
        "train_f1": round(train_f1, 4),
        "test_accuracy": round(accuracy_score(y_test, test_predictions), 4),
        "test_precision": round(precision_score(y_test, test_predictions, zero_division=0), 4),
        "test_recall": round(recall_score(y_test, test_predictions, zero_division=0), 4),
        "test_f1": round(test_f1, 4),
        "test_roc_auc": round(roc_auc, 4) if not pd.isna(roc_auc) else np.nan,
        "train_test_f1_gap": round(train_test_gap, 4),
        "overfitting_warning": overfitting_warning,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp)
    }

    return result


def evaluate_models(df):
    feature_df, target, numeric_features, categorical_features, excluded_present = prepare_features(df)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_df,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    models = get_models()

    results = []

    for model_name, model in models.items():
        result = evaluate_single_model(
            model_name=model_name,
            model=model,
            preprocessor=preprocessor,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test
        )

        results.append(result)

    results_df = pd.DataFrame(results)

    metadata = {
        "total_rows": len(df),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "target_column": TARGET_COLUMN,
        "excluded_columns": excluded_present,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features
    }

    return results_df, metadata


def generate_model_report(results_df, metadata, output_path):
    output = []

    best_model = results_df.sort_values(
        by="test_f1",
        ascending=False
    ).iloc[0]

    output.append("# Baseline Model Evaluation Report")
    output.append("")

    output.append("## Dataset Split")
    output.append("")
    output.append(f"- Total rows: {metadata['total_rows']}")
    output.append(f"- Training rows: {metadata['train_rows']}")
    output.append(f"- Testing rows: {metadata['test_rows']}")
    output.append(f"- Target column: `{metadata['target_column']}`")
    output.append(f"- Excluded risky columns: `{metadata['excluded_columns']}`")

    output.append("")
    output.append("## Feature Set")
    output.append("")
    output.append(f"- Numeric features: `{metadata['numeric_features']}`")
    output.append(f"- Categorical features: `{metadata['categorical_features']}`")

    output.append("")
    output.append("## Model Comparison")
    output.append("")
    output.append(results_df.to_markdown(index=False))

    output.append("")
    output.append("## Best Baseline Model")
    output.append("")
    output.append(f"- Best model by test F1-score: **{best_model['model_name']}**")
    output.append(f"- Test F1-score: **{best_model['test_f1']}**")
    output.append(f"- Test ROC-AUC: **{best_model['test_roc_auc']}**")
    output.append(f"- Overfitting warning: **{best_model['overfitting_warning']}**")

    output.append("")
    output.append("## Interpretation")
    output.append("")
    output.append("- This is a baseline model evaluation, not a final production model.")
    output.append("- Identifier columns and suspicious post-event columns were excluded before training.")
    output.append("- Missing values were imputed using median or most frequent strategy.")
    output.append("- Categorical columns were one-hot encoded.")
    output.append("- If recall is low, the model may miss positive churn cases.")
    output.append("- If train-test F1 gap is high, the model may be overfitting.")
    output.append("- Final model training should happen only after data cleaning, leakage review, drift monitoring, and fairness checks.")

    output.append("")
    output.append("## Required Next Action")
    output.append("")
    output.append("1. Clean invalid negative values in age and monthly_spend.")
    output.append("2. Normalize inconsistent categories such as Male/male and F/Female.")
    output.append("3. Re-run data quality and AI readiness audits after cleaning.")
    output.append("4. Compare model performance before and after cleaning.")
    output.append("5. Continue with LLM response evaluation module after baseline model evaluation.")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    input_file = "data/raw/sample_customer_ai_audit_dataset.csv"

    df = load_dataset(input_file)

    results_df, metadata = evaluate_models(df)

    Path("data/exports").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        "data/exports/model_evaluation_results.csv",
        index=False
    )

    generate_model_report(
        results_df=results_df,
        metadata=metadata,
        output_path="reports/model_evaluation_report.md"
    )

    best_model = results_df.sort_values(
        by="test_f1",
        ascending=False
    ).iloc[0]

    print("Baseline model evaluation completed.")
    print(f"Best model: {best_model['model_name']}")
    print(f"Best test F1-score: {best_model['test_f1']}")
    print(f"Best ROC-AUC: {best_model['test_roc_auc']}")
    print("Report saved: reports/model_evaluation_report.md")
    print("Results saved: data/exports/model_evaluation_results.csv")