"""Streamlit dashboard: churn analysis, model performance and customer prediction."""
import json

import pandas as pd
import streamlit as st

import plots
from predict import (
    explain_customer,
    load_model,
    load_model_info,
    predict_customer,
    validate_customer,
    version_warning,
)
from preprocessing import (
    CATEGORICAL_FEATURES,
    CATEGORY_OPTIONS,
    DATA_PATH,
    ROOT,
    TARGET,
    clean_data,
    load_raw,
)

REPORT_DIR = ROOT / "reports"

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1150px;}
    h1, h2, h3 {font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading dataset...")
def get_data() -> pd.DataFrame:
    return clean_data(load_raw())  # downloads the CSV on first run


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None


# --------------------------------------------------------------------------
# Page: churn analysis
# --------------------------------------------------------------------------
def page_analysis(df: pd.DataFrame) -> None:
    st.header("Churn analysis")
    st.caption("All charts are computed from the dataset and respond to the filters.")

    f1, f2, f3 = st.columns(3)
    contracts = f1.multiselect("Contract", CATEGORY_OPTIONS["Contract"],
                               default=CATEGORY_OPTIONS["Contract"])
    internet = f2.multiselect("Internet service", CATEGORY_OPTIONS["InternetService"],
                              default=CATEGORY_OPTIONS["InternetService"])
    tenure_range = f3.slider("Tenure (months)", int(df["tenure"].min()),
                             int(df["tenure"].max()),
                             (int(df["tenure"].min()), int(df["tenure"].max())))

    view = df[
        df["Contract"].isin(contracts)
        & df["InternetService"].isin(internet)
        & df["tenure"].between(*tenure_range)
    ]
    if view.empty:
        st.warning("No customers match the selected filters.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers", f"{len(view):,}")
    m2.metric("Churn rate", f"{view[TARGET].mean():.1%}")
    m3.metric("Average tenure", f"{view['tenure'].mean():.1f} months")
    m4.metric("Average monthly charge", f"${view['MonthlyCharges'].mean():.2f}")

    left, right = st.columns(2)
    left.pyplot(plots.churn_distribution(view))
    right.pyplot(plots.churn_rate_by_tenure_group(view))

    left, right = st.columns(2)
    left.pyplot(plots.numeric_hist_by_churn(view, "MonthlyCharges"))
    right.pyplot(plots.numeric_hist_by_churn(view, "TotalCharges"))

    st.subheader("Churn rate by customer characteristic")
    default_index = CATEGORICAL_FEATURES.index("Contract")
    column = st.selectbox("Characteristic", CATEGORICAL_FEATURES, index=default_index)
    st.pyplot(plots.churn_rate_by_category(view, column))

    # Plain statement computed from the filtered data (groups with 30+ customers).
    stats = view.groupby(column)[TARGET].agg(["mean", "count"])
    stats = stats[stats["count"] >= 30].sort_values("mean")
    if len(stats) >= 2:
        low, high = stats.index[0], stats.index[-1]
        st.write(
            f"In the filtered data, '{high}' has the highest churn rate "
            f"({stats['mean'].iloc[-1]:.1%}) and '{low}' the lowest "
            f"({stats['mean'].iloc[0]:.1%}). These are associations in past data, "
            "not proof that the characteristic causes churn."
        )


# --------------------------------------------------------------------------
# Page: model performance
# --------------------------------------------------------------------------
def page_performance(info: dict) -> None:
    st.header("Model performance")
    metrics = read_json(REPORT_DIR / "test_metrics.json")
    comparison_path = REPORT_DIR / "model_comparison.csv"
    if metrics is None or not comparison_path.exists():
        st.info("No evaluation results found. Run `python train.py` and `python evaluate.py`.")
        return

    st.subheader("Model comparison (5-fold cross-validation on training data)")
    comparison = pd.read_csv(comparison_path)
    comparison.columns = [c.replace("cv_", "").replace("roc_auc", "ROC-AUC").title()
                          if c != "model" else "Model" for c in comparison.columns]
    st.dataframe(comparison.round(3), hide_index=True, use_container_width=True)
    st.caption(f"Selected model: {metrics['model_name']} "
               f"(highest cross-validated F1 for the churn class).")

    st.subheader("Selected model on the held-out test set")
    cols = st.columns(5)
    for col, (label, key) in zip(
        cols,
        [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
         ("F1-score", "f1"), ("ROC-AUC", "roc_auc")],
    ):
        col.metric(label, f"{metrics[key]:.3f}")
    st.caption(
        f"{metrics['test_samples']:,} test customers, {metrics['test_churn_rate']:.1%} churned. "
        f"Always predicting 'stay' would score {metrics['majority_class_accuracy']:.1%} accuracy, "
        "so accuracy alone says little here."
    )

    st.markdown(
        "**Precision** is the share of customers flagged as churners who really left. "
        "**Recall** is the share of real churners the model catches. Missing a churner "
        "(low recall) loses revenue; flagging loyal customers (low precision) wastes "
        "retention offers. F1 balances the two, so it was used to choose the model."
    )

    left, right = st.columns(2)
    if (REPORT_DIR / "confusion_matrix.png").exists():
        left.image(str(REPORT_DIR / "confusion_matrix.png"))
    if (REPORT_DIR / "roc_curve.png").exists():
        right.image(str(REPORT_DIR / "roc_curve.png"))

    importance_path = REPORT_DIR / "permutation_importance.csv"
    if importance_path.exists():
        st.subheader("Features the model relies on")
        st.image(str(REPORT_DIR / "permutation_importance.png"))
        st.caption("Permutation importance shows which inputs the model uses. "
                   "It does not show that they cause churn.")


# --------------------------------------------------------------------------
# Page: customer prediction
# --------------------------------------------------------------------------
def page_prediction(model, df: pd.DataFrame) -> None:
    st.header("Predict churn for a customer")
    st.write("Enter the customer's details. The estimate comes from the trained model.")

    def choice(column, label, container, default=None):
        options = CATEGORY_OPTIONS[column]
        index = options.index(default) if default in options else 0
        return container.selectbox(label, options, index=index, key=column)

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)
        inputs = {}
        inputs["tenure"] = c1.number_input("Tenure (months)", 0, 120, 12)
        inputs["MonthlyCharges"] = c1.number_input("Monthly charges ($)", 0.0, 500.0, 70.0, 0.5)
        estimate_total = c1.checkbox("Estimate total charges (tenure x monthly)", value=True)
        total_input = c1.number_input("Total charges ($)", 0.0, 100000.0, 800.0, 10.0)
        inputs["Contract"] = choice("Contract", "Contract", c2, "Month-to-month")
        inputs["PaymentMethod"] = choice("PaymentMethod", "Payment method", c2, "Electronic check")
        inputs["PaperlessBilling"] = choice("PaperlessBilling", "Paperless billing", c2, "Yes")
        inputs["InternetService"] = choice("InternetService", "Internet service", c2, "Fiber optic")
        inputs["PhoneService"] = choice("PhoneService", "Phone service", c2, "Yes")
        inputs["MultipleLines"] = choice("MultipleLines", "Multiple lines", c2, "No")
        for col, label in [("OnlineSecurity", "Online security"), ("OnlineBackup", "Online backup"),
                           ("DeviceProtection", "Device protection"), ("TechSupport", "Tech support"),
                           ("StreamingTV", "Streaming TV"), ("StreamingMovies", "Streaming movies")]:
            inputs[col] = choice(col, label, c3, "No")
        inputs["gender"] = choice("gender", "Gender", c3)
        inputs["SeniorCitizen"] = choice("SeniorCitizen", "Senior citizen", c3, "No")
        inputs["Partner"] = choice("Partner", "Partner", c3, "No")
        inputs["Dependents"] = choice("Dependents", "Dependents", c3, "No")
        submitted = st.form_submit_button("Predict")

    st.caption("If internet service is 'No', add-on services are set to 'No internet service'. "
               "If phone service is 'No', multiple lines is set to 'No phone service'.")
    if not submitted:
        return

    inputs["TotalCharges"] = inputs["tenure"] * inputs["MonthlyCharges"] if estimate_total else total_input
    try:
        row = validate_customer(inputs)
        result = predict_customer(model, row)
        effects = explain_customer(model, row, df[row.columns])
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        return
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    st.subheader("Result")
    r1, r2 = st.columns(2)
    r1.metric("Predicted class", result["text"])
    r2.metric("Estimated churn probability", f"{result['probability']:.1%}")
    st.progress(min(max(result["probability"], 0.0), 1.0))
    st.caption("This probability is a model estimate based on past customers. "
               "It is not a guarantee about this customer.")

    st.subheader("Factors behind this estimate")
    if effects.empty:
        st.write("No single feature changes the estimate when set to its typical value.")
    else:
        top = effects.head(8)
        st.pyplot(plots.effect_bar(top))
        st.caption(
            "Each bar shows how the estimate changes if that one value were replaced "
            "by the typical value in the dataset (median or most common). Orange raises "
            "the churn estimate, blue lowers it. This explains the model, not the cause of churn."
        )


# --------------------------------------------------------------------------
# Page: project info
# --------------------------------------------------------------------------
def page_info(info: dict, df: pd.DataFrame) -> None:
    st.header("Project information")
    st.write(
        "This project predicts whether a subscription customer is likely to leave, "
        "compares three classification models and explains the results in a dashboard."
    )
    st.subheader("Dataset")
    st.write(
        f"IBM Telco Customer Churn sample dataset: {len(df):,} customers after cleaning, "
        f"{df.shape[1]} columns (the dataset file is stored at `{DATA_PATH.relative_to(ROOT)}`). "
        "The data is a fictional sample published by IBM for demonstration."
    )
    st.subheader("Method")
    st.write(
        "Numeric features are median-imputed and scaled; categorical features are "
        "one-hot encoded. Three models are compared with 5-fold cross-validation on "
        "the training split, the best by F1 is refitted, and the held-out test set is "
        "used once for the final metrics."
    )
    if info:
        st.write(f"Trained model: **{info.get('model_name')}** "
                 f"(trained {info.get('trained_at_utc', 'unknown')}, "
                 f"scikit-learn {info.get('sklearn_version', '?')}).")
    st.subheader("Limitations")
    st.write(
        "- The data comes from one telecom sample and may not match other businesses.\n"
        "- The model finds associations in past data, not causes of churn.\n"
        "- Probabilities are estimates and are not calibrated for any specific company.\n"
        "- The data has no information about time, so it cannot show trends."
    )


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    st.title("Customer Churn Prediction")
    page = st.sidebar.radio(
        "Navigation",
        ["Customer prediction", "Churn analysis", "Model performance", "Project info"],
    )

    try:
        df = get_data()
    except Exception as exc:
        st.error(f"Dataset unavailable. {exc}")
        st.stop()

    info = load_model_info()
    if page == "Churn analysis":
        page_analysis(df)
    elif page == "Project info":
        page_info(info, df)
    else:
        try:
            model = get_model()
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        warning = version_warning(info)
        if warning:
            st.warning(warning)
        if page == "Model performance":
            page_performance(info)
        else:
            page_prediction(model, df)


main()
