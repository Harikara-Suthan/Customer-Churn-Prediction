"""Dataset download, cleaning, train/test split and the preprocessing step.

Dataset: IBM Telco Customer Churn (sample dataset published by IBM).
Every other module imports its constants from here so that training and
inference always use the same column names and the same column order.
"""
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "telco_churn.csv"

# Public mirrors of the IBM Telco Customer Churn CSV (tried in order).
DATA_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]

TARGET = "Churn"  # "Yes"/"No" in the CSV, mapped to 1/0 in clean_data
RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

# Allowed values for every categorical feature (taken from the dataset schema).
CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": ["No", "Yes"],
    "Partner": ["No", "Yes"],
    "Dependents": ["No", "Yes"],
    "PhoneService": ["No", "Yes"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["No", "Yes"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}
CATEGORICAL_FEATURES = list(CATEGORY_OPTIONS)
# One fixed column order for training and inference.
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

MANUAL_DOWNLOAD_HELP = (
    "Could not download the dataset automatically. Download "
    "'WA_Fn-UseC_-Telco-Customer-Churn.csv' from "
    "https://www.kaggle.com/datasets/blastchar/telco-customer-churn "
    f"and save it as {DATA_PATH}."
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _looks_like_telco_csv(path: Path) -> bool:
    """Quick check that a file has the expected header."""
    try:
        header = pd.read_csv(path, nrows=0).columns.str.strip()
    except Exception:
        return False
    return all(col in header for col in FEATURES + [TARGET])


def download_dataset(path: Path = DATA_PATH, urls=DATA_URLS) -> Path:
    """Download the CSV once. Does nothing if a valid file already exists."""
    path = Path(path)
    if path.exists() and _looks_like_telco_csv(path):
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    errors = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=30) as response, open(tmp_path, "wb") as out:
                shutil.copyfileobj(response, out)
            if not _looks_like_telco_csv(tmp_path):
                raise ValueError("downloaded file does not have the expected columns")
            tmp_path.replace(path)
            return path
        except Exception as exc:  # network error, bad file, etc.
            errors.append(f"{url}: {exc}")
    tmp_path.unlink(missing_ok=True)
    raise RuntimeError(MANUAL_DOWNLOAD_HELP + "\nDetails:\n" + "\n".join(errors))


def load_raw(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV (downloading it first if needed)."""
    return pd.read_csv(download_dataset(path))


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------
def describe_quality(raw: pd.DataFrame) -> dict:
    """Dimensions, dtypes, missing values, blank strings and duplicates."""
    blanks = {}
    for col in raw.columns:
        if not pd.api.types.is_numeric_dtype(raw[col]):
            count = int((raw[col].astype(str).str.strip() == "").sum())
            if count:
                blanks[col] = count
    missing = raw.isna().sum()
    return {
        "rows": int(raw.shape[0]),
        "columns": int(raw.shape[1]),
        "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
        "missing_values": {c: int(n) for c, n in missing.items() if n},
        "blank_strings": blanks,
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_customer_ids": int(raw["customerID"].duplicated().sum())
        if "customerID" in raw.columns
        else None,
    }


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with a numeric 0/1 target.

    - TotalCharges has blank strings in the raw file; they become NaN.
      For customers with tenure 0 (not billed yet) the value is set to 0.
    - Negative or non-numeric values in numeric columns become NaN and are
      imputed later inside the pipeline (so the imputer is fitted on train only).
    - Rows with an unknown target label are dropped.
    """
    df = raw.copy()
    df.columns = df.columns.str.strip()

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().astype(object)

    df = df.drop_duplicates().reset_index(drop=True)

    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan
    df.loc[(df["tenure"] == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0.0

    # SeniorCitizen is 0/1 in the CSV; use Yes/No like the other flags.
    df["SeniorCitizen"] = df["SeniorCitizen"].map(
        {0: "No", 1: "Yes", "0": "No", "1": "Yes", "No": "No", "Yes": "Yes"}
    )

    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df[TARGET] = df[TARGET].astype(int)
    return df


# --------------------------------------------------------------------------
# Split and preprocessing
# --------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    """Stratified train/test split. The same seed gives the same split everywhere."""
    X = df[FEATURES]
    y = df[TARGET]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical ones.

    This object is part of the saved pipeline, so it is fitted on training
    data only and reused unchanged at prediction time.
    """
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ]
    )


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
