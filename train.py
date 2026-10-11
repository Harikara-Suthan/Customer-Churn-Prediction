"""Compare three classifiers with cross-validation and save the best pipeline.

Only the training split is used here. The held-out test set is touched
once, in evaluate.py, after the final model has been chosen.
"""
import platform
from datetime import datetime, timezone

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from preprocessing import (
    FEATURES,
    RANDOM_STATE,
    build_preprocessor,
    clean_data,
    describe_quality,
    load_raw,
    save_json,
    split_data,
    ROOT,
)

MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
MODEL_INFO_PATH = ROOT / "models" / "model_info.json"
REPORT_DIR = ROOT / "reports"

# Churn is the minority class, so every model uses class weights.
SELECTION_METRIC = "f1"


def get_models() -> dict:
    """Each model is a full pipeline: preprocessing + classifier."""
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_depth=3,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    }
    return {
        name: Pipeline([("preprocess", build_preprocessor()), ("model", clf)])
        for name, clf in classifiers.items()
    }


def compare_models(models: dict, X_train, y_train) -> pd.DataFrame:
    """5-fold stratified cross-validation on the training split."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []
    for name, pipeline in models.items():
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        row = {"model": name}
        for metric in scoring:
            row[f"cv_{metric}"] = scores[f"test_{metric}"].mean()
        rows.append(row)
        print(f"  finished {name}")
    return pd.DataFrame(rows).sort_values(f"cv_{SELECTION_METRIC}", ascending=False)


def main():
    raw = load_raw()
    quality = describe_quality(raw)
    print(f"Dataset: {quality['rows']} rows, {quality['columns']} columns")
    print(f"Missing values: {quality['missing_values']}  Blank strings: {quality['blank_strings']}")
    print(f"Duplicate rows: {quality['duplicate_rows']}")
    save_json(quality, REPORT_DIR / "data_quality.json")

    df = clean_data(raw)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"Train: {len(X_train)}  Test (not used here): {len(X_test)}")
    print(f"Churn rate in train: {y_train.mean():.3f}")

    models = get_models()
    print("Cross-validating models...")
    comparison = compare_models(models, X_train, y_train)
    print(comparison.round(4).to_string(index=False))

    REPORT_DIR.mkdir(exist_ok=True)
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    best_name = comparison.iloc[0]["model"]
    print(f"Selected model: {best_name} (highest cross-validated {SELECTION_METRIC})")
    final_model = clone(models[best_name]).fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    save_json(
        {
            "model_name": best_name,
            "selection_metric": f"cross-validated {SELECTION_METRIC}",
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "random_state": RANDOM_STATE,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": FEATURES,
        },
        MODEL_INFO_PATH,
    )
    print(f"Saved {MODEL_PATH}")
    print("Next: python evaluate.py")


if __name__ == "__main__":
    main()
