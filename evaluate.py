"""Evaluate the saved model once on the held-out test set.

Writes to reports/: test_metrics.json, confusion_matrix.png, roc_curve.png,
permutation_importance.csv/.png and exploratory charts (eda_*.png).
"""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import plots
from predict import THRESHOLD, churn_probability, load_model, load_model_info
from preprocessing import (
    CATEGORICAL_FEATURES,
    ROOT,
    TARGET,
    clean_data,
    load_raw,
    save_json,
    split_data,
)

REPORT_DIR = ROOT / "reports"


def save_figure(fig, name: str) -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    fig.savefig(REPORT_DIR / name, dpi=150)


def save_eda_figures(df: pd.DataFrame) -> None:
    """Static copies of the dashboard's main exploratory charts."""
    save_figure(plots.churn_distribution(df), "eda_churn_distribution.png")
    save_figure(plots.churn_rate_by_tenure_group(df), "eda_tenure.png")
    save_figure(plots.numeric_hist_by_churn(df, "MonthlyCharges"), "eda_monthly_charges.png")
    for column in ["Contract", "PaymentMethod", "InternetService"]:
        save_figure(plots.churn_rate_by_category(df, column), f"eda_{column}.png")


def main():
    pipeline = load_model()
    info = load_model_info()
    df = clean_data(load_raw())
    _, X_test, _, y_test = split_data(df)

    probabilities = churn_probability(pipeline, X_test)
    predictions = (probabilities >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    metrics = {
        "model_name": info.get("model_name", type(pipeline.steps[-1][1]).__name__),
        "threshold": THRESHOLD,
        "test_samples": int(len(y_test)),
        "test_churn_rate": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        # Accuracy of always predicting "stay": the baseline accuracy must beat.
        "majority_class_accuracy": float(1 - y_test.mean()),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    save_json(metrics, REPORT_DIR / "test_metrics.json")
    for name in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{name:>10}: {metrics[name]:.4f}")
    print(f"Confusion matrix  TN={tn} FP={fp} FN={fn} TP={tp}")
    print(f"Majority-class baseline accuracy: {metrics['majority_class_accuracy']:.4f}")

    save_figure(plots.confusion_matrix_fig(cm), "confusion_matrix.png")
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    save_figure(plots.roc_curve_fig(fpr, tpr, metrics["roc_auc"]), "roc_curve.png")

    # Permutation importance: drop in ROC-AUC when one feature is shuffled.
    result = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42
    )
    importance = (
        pd.DataFrame({"feature": X_test.columns, "value": result.importances_mean,
                      "std": result.importances_std})
        .sort_values("value", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    save_figure(
        plots.importance_bar(
            importance.head(10),
            "Permutation importance (test set)",
            "Drop in ROC-AUC when the feature is shuffled",
        ),
        "permutation_importance.png",
    )

    save_eda_figures(df)
    print(f"Saved reports to {REPORT_DIR}")


if __name__ == "__main__":
    main()
