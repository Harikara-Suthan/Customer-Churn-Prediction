"""Shared inference helpers used by the Streamlit app, evaluate.py and tests."""
import json

import joblib
import numpy as np
import pandas as pd
import sklearn

from preprocessing import (
    CATEGORICAL_FEATURES,
    CATEGORY_OPTIONS,
    FEATURES,
    NUMERIC_FEATURES,
    ROOT,
)

MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
MODEL_INFO_PATH = ROOT / "models" / "model_info.json"
THRESHOLD = 0.5  # probability >= threshold -> predicted "churn"
CLASS_LABELS = {0: "Will stay", 1: "Likely to churn"}

NUMERIC_RANGES = {
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 100000),
}
ADD_ON_SERVICES = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file does not exist."""


def load_model(path=MODEL_PATH):
    """Load the saved pipeline. Raises a clear error if it is missing or unreadable."""
    if not path.exists():
        raise ModelNotFoundError(
            f"Model file not found: {path}. Run `python train.py` first."
        )
    try:
        return joblib.load(path)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load the model ({exc}). It may have been saved with a "
            "different scikit-learn version; run `python train.py` again."
        ) from exc


def load_model_info(path=MODEL_INFO_PATH) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def version_warning(info: dict):
    """Return a warning string if the model was trained with another sklearn version."""
    trained = info.get("sklearn_version")
    if trained and trained != sklearn.__version__:
        return (
            f"The model was trained with scikit-learn {trained} but "
            f"{sklearn.__version__} is installed. Retrain if predictions fail."
        )
    return None


def normalize_services(inputs: dict) -> dict:
    """Keep related service fields consistent, as they are in the dataset."""
    fixed = dict(inputs)
    if fixed.get("InternetService") == "No":
        for col in ADD_ON_SERVICES:
            fixed[col] = "No internet service"
    if fixed.get("PhoneService") == "No":
        fixed["MultipleLines"] = "No phone service"
    return fixed


def validate_customer(inputs: dict) -> pd.DataFrame:
    """Check a customer dict and return a one-row DataFrame in training column order.

    Raises ValueError listing every problem found.
    """
    problems = []
    data = {}
    inputs = normalize_services(inputs)

    for col in CATEGORICAL_FEATURES:
        value = inputs.get(col)
        if value not in CATEGORY_OPTIONS[col]:
            problems.append(f"{col}: '{value}' is not one of {CATEGORY_OPTIONS[col]}")
        data[col] = value

    for col in NUMERIC_FEATURES:
        low, high = NUMERIC_RANGES[col]
        try:
            value = float(inputs.get(col))
        except (TypeError, ValueError):
            problems.append(f"{col}: must be a number")
            continue
        if not np.isfinite(value) or not low <= value <= high:
            problems.append(f"{col}: must be between {low} and {high}")
        data[col] = value

    if problems:
        raise ValueError("; ".join(problems))
    return pd.DataFrame([data], columns=FEATURES)


def churn_probability(pipeline, X: pd.DataFrame) -> np.ndarray:
    """Probability of class 1 (churn) for each row, using the model's own class order."""
    churn_column = list(pipeline.classes_).index(1)
    return pipeline.predict_proba(X)[:, churn_column]


def predict_customer(pipeline, row: pd.DataFrame) -> dict:
    probability = float(churn_probability(pipeline, row)[0])
    label = int(probability >= THRESHOLD)
    return {"probability": probability, "label": label, "text": CLASS_LABELS[label]}


def explain_customer(pipeline, row: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Single-feature what-if analysis.

    For each feature, replace this customer's value with the typical value
    (median or most common) and measure how the churn probability changes.
    A positive change means the customer's value pushes the estimate up
    compared with a typical customer. This describes the model's behaviour,
    not causes of churn.
    """
    base = float(churn_probability(pipeline, row)[0])
    typical = {}
    for col in NUMERIC_FEATURES:
        typical[col] = float(reference[col].median())
    for col in CATEGORICAL_FEATURES:
        typical[col] = reference[col].mode().iloc[0]

    variants = []
    for col in FEATURES:
        variant = row.copy()
        variant[col] = typical[col]
        variants.append(variant)
    changed = churn_probability(pipeline, pd.concat(variants, ignore_index=True))

    result = pd.DataFrame(
        {
            "feature": FEATURES,
            "customer_value": [row.iloc[0][c] for c in FEATURES],
            "typical_value": [typical[c] for c in FEATURES],
            "effect_on_probability": base - changed,  # in probability points (0-1)
        }
    )
    result = result[result["effect_on_probability"].abs() > 1e-6]
    order = result["effect_on_probability"].abs().sort_values(ascending=False).index
    return result.loc[order].reset_index(drop=True)
