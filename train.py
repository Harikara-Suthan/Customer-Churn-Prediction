
"""Compare churn classifiers and save the best pipeline.

Model selection uses stratified cross-validation on training data only.
The held-out test set is reserved for evaluate.py.
"""

import platform
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from preprocessing import (
    FEATURES,
    RANDOM_STATE,
    ROOT,
    build_preprocessor,
    clean_data,
    describe_quality,
    load_raw,
    save_json,
    split_data,
)

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
MODEL_INFO_PATH = MODEL_DIR / "model_info.json"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"

# Accuracy is the primary selection metric for this training run.
SELECTION_METRIC = "accuracy"


def get_models():
    """Create complete preprocessing and classifier pipelines."""
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=2,
            min_samples_leaf=10,
            random_state=RANDOM_STATE,
        ),
    }

    return {
        name: Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", classifier),
            ]
        )
        for name, classifier in classifiers.items()
    }


def compare_models(models, X_train, y_train):
    """Compare models using stratified five-fold cross-validation."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    rows = []

    for name, pipeline in models.items():
        print(f"\nTraining and cross-validating: {name}")

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"cv_{metric}"] = float(values.mean())
            row[f"std_{metric}"] = float(values.std())

        rows.append(row)

        print(
            f"  Accuracy: {row['cv_accuracy']:.4f} "
            f"+/- {row['std_accuracy']:.4f}"
        )
        print(
            f"  Precision: {row['cv_precision']:.4f} | "
            f"Recall: {row['cv_recall']:.4f} | "
            f"F1: {row['cv_f1']:.4f} | "
            f"ROC-AUC: {row['cv_roc_auc']:.4f}"
        )

    comparison = pd.DataFrame(rows)

    # Use ROC-AUC as a tie-breaker when CV accuracy is equal.
    comparison = comparison.sort_values(
        by=["cv_accuracy", "cv_roc_auc"],
        ascending=False,
    ).reset_index(drop=True)

    return comparison


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CUSTOMER CHURN MODEL TRAINING")
    print("=" * 60)

    raw = load_raw()
    quality = describe_quality(raw)

    print(
        f"\nDataset: {quality['rows']} rows, "
        f"{quality['columns']} columns"
    )
    print(f"Missing values: {quality['missing_values']}")
    print(f"Blank strings: {quality['blank_strings']}")
    print(f"Duplicate rows: {quality['duplicate_rows']}")

    save_json(quality, REPORT_DIR / "data_quality.json")

    df = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(df)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Held-out test samples: {len(X_test)}")
    print(f"Training churn rate: {y_train.mean():.2%}")

    if y_train.nunique() != 2:
        raise ValueError("Training data must contain both target classes.")

    models = get_models()

    print("\nStarting five-fold cross-validation...")
    comparison = compare_models(models, X_train, y_train)

    comparison.to_csv(COMPARISON_PATH, index=False)

    print("\nMODEL COMPARISON")
    print(
        comparison[
            [
                "model",
                "cv_accuracy",
                "cv_precision",
                "cv_recall",
                "cv_f1",
                "cv_roc_auc",
            ]
        ].round(4).to_string(index=False)
    )

    # Select using cross-validation, never using held-out test results.
    best_name = comparison.iloc[0]["model"]
    best_row = comparison.iloc[0]
    best_cv_accuracy = float(best_row["cv_accuracy"])

    print(f"\nSelected model: {best_name}")
    print(f"Mean CV accuracy: {best_cv_accuracy:.2%}")
    print("Fitting selected pipeline on all training samples...")

    final_model = clone(models[best_name])
    final_model.fit(X_train, y_train)

    joblib.dump(final_model, MODEL_PATH)

    model_info = {
        "model_name": best_name,
        "selection_metric": "5-fold cross-validated accuracy",
        "cross_validated_accuracy": best_cv_accuracy,
        "cross_validated_accuracy_std": float(
            best_row["std_accuracy"]
        ),
        "cross_validated_precision": float(best_row["cv_precision"]),
        "cross_validated_recall": float(best_row["cv_recall"]),
        "cross_validated_f1": float(best_row["cv_f1"]),
        "cross_validated_roc_auc": float(best_row["cv_roc_auc"]),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "python_version": platform.python_version(),
        "scikit_learn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "cv_folds": 5,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(FEATURES),
    }

    save_json(model_info, MODEL_INFO_PATH)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model: {best_name}")
    print(f"Cross-validation accuracy: {best_cv_accuracy:.2%}")
    print(f"Saved pipeline: {MODEL_PATH}")
    print(f"Saved model information: {MODEL_INFO_PATH}")
    print(f"Saved comparison: {COMPARISON_PATH}")
    print("\nNext command: python evaluate.py")
    print("The held-out test set was not used for model selection.")


if __name__ == "__main__":
    main()
