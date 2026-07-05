import inspect
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)


TARGET_COLUMN = "churn"

EXCLUDED_COLUMNS = [
    "customer_id",
    "last_payment_status"
]

MAX_TRAINING_ROWS = 50000
RANDOM_STATE = 42


def make_one_hot_encoder():
    params = inspect.signature(OneHotEncoder).parameters

    if "sparse_output" in params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(input_path):
    df = pd.read_csv(input_path)

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    return df


def normalize_target(target):
    unique_values = sorted(pd.Series(target).dropna().unique().tolist())

    if len(unique_values) != 2:
        raise ValueError(
            "This production evaluator currently supports binary classification only. "
            f"Found target values: {unique_values}"
        )

    return target


def prepare_features(df):
    excluded_present = [
        column for column in EXCLUDED_COLUMNS
        if column in df.columns
    ]

    feature_df = df.drop(
        columns=[TARGET_COLUMN] + excluded_present,
        errors="ignore"
    )

    target = normalize_target(df[TARGET_COLUMN])

    numeric_features = feature_df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = feature_df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()

    ignored_features = [
        column for column in feature_df.columns
        if column not in numeric_features + categorical_features
    ]

    return (
        feature_df,
        target,
        numeric_features,
        categorical_features,
        ignored_features,
        excluded_present,
    )


def apply_large_dataset_guard(feature_df, target):
    if len(feature_df) <= MAX_TRAINING_ROWS:
        return feature_df, target, False

    sampled_df = feature_df.copy()
    sampled_df[TARGET_COLUMN] = target.values

    sampled_df = sampled_df.groupby(
        TARGET_COLUMN,
        group_keys=False
    ).apply(
        lambda group: group.sample(
            min(
                len(group),
                max(1, int(MAX_TRAINING_ROWS * len(group) / len(sampled_df)))
            ),
            random_state=RANDOM_STATE
        )
    )

    sampled_target = sampled_df[TARGET_COLUMN]
    sampled_features = sampled_df.drop(columns=[TARGET_COLUMN])

    return sampled_features, sampled_target, True


def build_preprocessor(numeric_features, categorical_features):
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_one_hot_encoder())
            ]
        )

        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    if not transformers:
        raise ValueError("No usable numeric or categorical features found.")

    return ColumnTransformer(transformers=transformers)


def get_models():
    return {
        "Dummy Baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_leaf_nodes=31,
            l2_regularization=0.1,
            random_state=RANDOM_STATE
        ),
    }


def safe_roc_auc(y_true, probabilities):
    try:
        return roc_auc_score(y_true, probabilities)
    except Exception:
        return np.nan


def safe_average_precision(y_true, probabilities):
    try:
        return average_precision_score(y_true, probabilities)
    except Exception:
        return np.nan


def get_probabilities(pipeline, x_test):
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(x_test)

        if probabilities.shape[1] >= 2:
            return probabilities[:, 1]

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(x_test)
        return 1 / (1 + np.exp(-scores))

    return None


def tune_threshold(y_test, probabilities):
    if probabilities is None:
        return {
            "best_threshold": 0.5,
            "best_threshold_f1": np.nan,
            "best_threshold_precision": np.nan,
            "best_threshold_recall": np.nan,
        }

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)

    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1": f1_score(y_test, predictions, zero_division=0),
        })

    threshold_df = pd.DataFrame(rows)

    best_row = threshold_df.sort_values(
        by="f1",
        ascending=False
    ).iloc[0]

    return {
        "best_threshold": round(float(best_row["threshold"]), 2),
        "best_threshold_f1": round(float(best_row["f1"]), 4),
        "best_threshold_precision": round(float(best_row["precision"]), 4),
        "best_threshold_recall": round(float(best_row["recall"]), 4),
    }


def calculate_cv_scores(pipeline, x_train, y_train):
    class_counts = pd.Series(y_train).value_counts()

    min_class_count = int(class_counts.min())

    if min_class_count < 3:
        return {
            "cv_f1_mean": np.nan,
            "cv_f1_std": np.nan,
            "cv_roc_auc_mean": np.nan,
            "cv_roc_auc_std": np.nan,
        }

    n_splits = min(5, min_class_count)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    try:
        f1_scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring="f1",
            n_jobs=-1
        )

        roc_scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        return {
            "cv_f1_mean": round(float(np.mean(f1_scores)), 4),
            "cv_f1_std": round(float(np.std(f1_scores)), 4),
            "cv_roc_auc_mean": round(float(np.mean(roc_scores)), 4),
            "cv_roc_auc_std": round(float(np.std(roc_scores)), 4),
        }

    except Exception:
        return {
            "cv_f1_mean": np.nan,
            "cv_f1_std": np.nan,
            "cv_roc_auc_mean": np.nan,
            "cv_roc_auc_std": np.nan,
        }


def overfitting_label(train_f1, test_f1):
    gap = train_f1 - test_f1

    if gap >= 0.20:
        return "High Overfitting Risk"

    if gap >= 0.10:
        return "Moderate Overfitting Risk"

    return "Low Overfitting Risk"


def evaluate_single_model(model_name, model, preprocessor, x_train, x_test, y_train, y_test):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    cv_scores = calculate_cv_scores(
        pipeline=pipeline,
        x_train=x_train,
        y_train=y_train
    )

    pipeline.fit(x_train, y_train)

    train_predictions = pipeline.predict(x_train)
    test_predictions = pipeline.predict(x_test)

    probabilities = get_probabilities(
        pipeline=pipeline,
        x_test=x_test
    )

    threshold_info = tune_threshold(
        y_test=y_test,
        probabilities=probabilities
    )

    train_f1 = f1_score(y_train, train_predictions, zero_division=0)
    test_f1 = f1_score(y_test, test_predictions, zero_division=0)

    cm = confusion_matrix(y_test, test_predictions)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    roc_auc = safe_roc_auc(y_test, probabilities) if probabilities is not None else np.nan
    avg_precision = safe_average_precision(y_test, probabilities) if probabilities is not None else np.nan

    result = {
        "model_name": model_name,
        "train_f1": round(train_f1, 4),
        "test_accuracy": round(accuracy_score(y_test, test_predictions), 4),
        "test_balanced_accuracy": round(balanced_accuracy_score(y_test, test_predictions), 4),
        "test_precision": round(precision_score(y_test, test_predictions, zero_division=0), 4),
        "test_recall": round(recall_score(y_test, test_predictions, zero_division=0), 4),
        "test_f1": round(test_f1, 4),
        "test_roc_auc": round(roc_auc, 4) if not pd.isna(roc_auc) else np.nan,
        "test_average_precision": round(avg_precision, 4) if not pd.isna(avg_precision) else np.nan,
        "cv_f1_mean": cv_scores["cv_f1_mean"],
        "cv_f1_std": cv_scores["cv_f1_std"],
        "cv_roc_auc_mean": cv_scores["cv_roc_auc_mean"],
        "cv_roc_auc_std": cv_scores["cv_roc_auc_std"],
        "best_threshold": threshold_info["best_threshold"],
        "best_threshold_f1": threshold_info["best_threshold_f1"],
        "best_threshold_precision": threshold_info["best_threshold_precision"],
        "best_threshold_recall": threshold_info["best_threshold_recall"],
        "train_test_f1_gap": round(train_f1 - test_f1, 4),
        "overfitting_warning": overfitting_label(train_f1, test_f1),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    return result, pipeline


def get_feature_names(preprocessor):
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def extract_feature_importance(best_pipeline, best_model_name):
    model = best_pipeline.named_steps["model"]
    preprocessor = best_pipeline.named_steps["preprocessor"]

    feature_names = get_feature_names(preprocessor)

    if not feature_names:
        return pd.DataFrame()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_

    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])

    else:
        return pd.DataFrame()

    if len(values) != len(feature_names):
        return pd.DataFrame()

    importance_df = pd.DataFrame({
        "model_name": best_model_name,
        "feature_name": feature_names,
        "importance": values
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    importance_df["importance"] = importance_df["importance"].round(6)

    return importance_df


def evaluate_models(df):
    (
        feature_df,
        target,
        numeric_features,
        categorical_features,
        ignored_features,
        excluded_present,
    ) = prepare_features(df)

    feature_df, target, sampled_for_training = apply_large_dataset_guard(
        feature_df=feature_df,
        target=target
    )

    x_train, x_test, y_train, y_test = train_test_split(
        feature_df,
        target,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=target
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    models = get_models()

    result_rows = []
    fitted_pipelines = {}

    for model_name, model in models.items():
        result, pipeline = evaluate_single_model(
            model_name=model_name,
            model=model,
            preprocessor=preprocessor,
            x_train=x_train,
            x_test=x_test,
            y_train=y_train,
            y_test=y_test
        )

        result_rows.append(result)
        fitted_pipelines[model_name] = pipeline

    results_df = pd.DataFrame(result_rows)

    best_model_row = results_df.sort_values(
        by=["test_f1", "test_roc_auc", "test_balanced_accuracy"],
        ascending=False
    ).iloc[0]

    best_model_name = best_model_row["model_name"]
    best_pipeline = fitted_pipelines[best_model_name]

    feature_importance_df = extract_feature_importance(
        best_pipeline=best_pipeline,
        best_model_name=best_model_name
    )

    threshold_analysis_df = results_df[
        [
            "model_name",
            "best_threshold",
            "best_threshold_f1",
            "best_threshold_precision",
            "best_threshold_recall"
        ]
    ].copy()

    metadata = {
        "total_rows": len(df),
        "training_rows_used": len(feature_df),
        "sampled_for_training": sampled_for_training,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "target_column": TARGET_COLUMN,
        "excluded_columns": excluded_present,
        "ignored_features": ignored_features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_distribution": target.value_counts(normalize=True).round(4).to_dict(),
        "best_model_name": best_model_name,
        "feature_importance_df": feature_importance_df,
        "threshold_analysis_df": threshold_analysis_df,
    }

    return results_df, metadata


def generate_model_report(results_df, metadata, output_path):
    best_model = results_df.sort_values(
        by=["test_f1", "test_roc_auc", "test_balanced_accuracy"],
        ascending=False
    ).iloc[0]

    feature_importance_df = metadata.get("feature_importance_df", pd.DataFrame())
    threshold_analysis_df = metadata.get("threshold_analysis_df", pd.DataFrame())

    output = []

    output.append("# Production Model Evaluation Report")
    output.append("")

    output.append("## Dataset Split")
    output.append("")
    output.append(f"- Total rows: {metadata['total_rows']}")
    output.append(f"- Training rows used after safety guard: {metadata['training_rows_used']}")
    output.append(f"- Large dataset sampling guard applied: `{metadata['sampled_for_training']}`")
    output.append(f"- Train rows: {metadata['train_rows']}")
    output.append(f"- Test rows: {metadata['test_rows']}")
    output.append(f"- Target column: `{metadata['target_column']}`")
    output.append(f"- Target distribution: `{metadata['target_distribution']}`")

    output.append("")
    output.append("## Feature Governance")
    output.append("")
    output.append(f"- Excluded risky columns: `{metadata['excluded_columns']}`")
    output.append(f"- Ignored unsupported features: `{metadata['ignored_features']}`")
    output.append(f"- Numeric features: `{metadata['numeric_features']}`")
    output.append(f"- Categorical features: `{metadata['categorical_features']}`")

    output.append("")
    output.append("## Model Leaderboard")
    output.append("")
    output.append(results_df.to_markdown(index=False))

    output.append("")
    output.append("## Best Model")
    output.append("")
    output.append(f"- Best model: **{best_model['model_name']}**")
    output.append(f"- Test F1-score: **{best_model['test_f1']}**")
    output.append(f"- Test ROC-AUC: **{best_model['test_roc_auc']}**")
    output.append(f"- Test average precision: **{best_model['test_average_precision']}**")
    output.append(f"- Best threshold: **{best_model['best_threshold']}**")
    output.append(f"- Best threshold F1-score: **{best_model['best_threshold_f1']}**")
    output.append(f"- Overfitting warning: **{best_model['overfitting_warning']}**")

    output.append("")
    output.append("## Threshold Analysis")
    output.append("")

    if threshold_analysis_df.empty:
        output.append("- Threshold analysis not available.")
    else:
        output.append(threshold_analysis_df.to_markdown(index=False))

    output.append("")
    output.append("## Top Feature Importance")
    output.append("")

    if feature_importance_df.empty:
        output.append("- Feature importance not available for the selected best model.")
    else:
        output.append(feature_importance_df.head(20).to_markdown(index=False))

    output.append("")
    output.append("## Production Interpretation")
    output.append("")
    output.append("- Dummy Baseline is included to prove whether ML models actually beat a naive baseline.")
    output.append("- Cross-validation metrics reduce the risk of trusting a lucky train-test split.")
    output.append("- Threshold tuning is included because default 0.50 classification threshold is often not optimal.")
    output.append("- Feature importance helps explain which signals drive predictions.")
    output.append("- This evaluator is stronger than a basic student ML script, but it is still not final MLOps.")
    output.append("- Production deployment would still require monitoring, retraining strategy, model registry, tests, and business approval.")

    output.append("")
    output.append("## Required Next Action")
    output.append("")
    output.append("1. Compare model performance against Dummy Baseline.")
    output.append("2. Investigate false negatives if recall is weak.")
    output.append("3. Review top features for leakage or proxy bias.")
    output.append("4. Use threshold tuning based on business cost, not only F1-score.")
    output.append("5. Re-run evaluation after data cleaning and feature governance.")

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

    metadata["threshold_analysis_df"].to_csv(
        "data/exports/model_threshold_analysis.csv",
        index=False
    )

    metadata["feature_importance_df"].to_csv(
        "data/exports/model_feature_importance.csv",
        index=False
    )

    generate_model_report(
        results_df=results_df,
        metadata=metadata,
        output_path="reports/model_evaluation_report.md"
    )

    best_model = results_df.sort_values(
        by=["test_f1", "test_roc_auc", "test_balanced_accuracy"],
        ascending=False
    ).iloc[0]

    print("Production model evaluation completed.")
    print(f"Best model: {best_model['model_name']}")
    print(f"Best F1-score: {best_model['test_f1']}")
    print(f"Best ROC-AUC: {best_model['test_roc_auc']}")
    print(f"Best threshold: {best_model['best_threshold']}")
    print("Report saved: reports/model_evaluation_report.md")
    print("Results saved: data/exports/model_evaluation_results.csv")
    print("Threshold analysis saved: data/exports/model_threshold_analysis.csv")
    print("Feature importance saved: data/exports/model_feature_importance.csv")