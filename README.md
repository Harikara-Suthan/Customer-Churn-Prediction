# Customer Churn Prediction

A machine learning project that predicts whether a subscription customer is likely to leave, compares three classification models, and presents the results in an interactive Streamlit dashboard.

## Problem statement
Keeping an existing customer is usually cheaper than winning a new one. Given a customer's contract, services, billing and tenure, can we estimate the probability that they will churn, and which characteristics are associated with churn?

## Objectives
- Analyse historical customer data and identify churn patterns.
- Train and compare Logistic Regression, Random Forest and Histogram Gradient Boosting.
- Estimate churn probability for an individual customer.
- Show evaluation metrics and an interpretable view of the model.

## Features
- Automatic dataset download, cleaning and validation
- Leakage-safe pipeline (imputation, scaling and encoding are fitted on training data only and saved with the model)
- 5-fold cross-validated model comparison, final test on a held-out set
- Dashboard pages: customer prediction, churn analysis (with filters), model performance, project info
- Input validation and clear error messages
- Automated tests (pytest)

## Dataset
**IBM Telco Customer Churn** sample dataset: 7,043 customers, 21 columns (demographics, services, contract, billing, and a `Churn` label). It is a fictional sample published by IBM for demonstration.

- Kaggle page: https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- Download used by the project (automatic): https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv

Data quality found by `train.py` (see `reports/data_quality.json`): no missing values, no duplicate rows, and 11 blank `TotalCharges` strings. All 11 belong to customers with tenure 0, so they are set to 0. In the cleaned data 26.5% of customers churned.

## Technologies
Python 3.12, pandas, NumPy, scikit-learn, matplotlib, joblib, Streamlit (exact versions in `requirements.txt`). pyarrow is pinned because the latest release needs NumPy 2, while this project uses NumPy 1.26.4.

## Workflow
1. **Cleaning** (`preprocessing.py`): strip whitespace, convert numeric columns, treat invalid values as missing, map `Churn` to 1/0, `SeniorCitizen` to Yes/No.
2. **Split**: 80/20 stratified train/test split with a fixed seed (42). The test set is not used for fitting or model selection.
3. **Preprocessing pipeline**: median imputation + standard scaling for numeric features; most-frequent imputation + one-hot encoding (unknown categories ignored) for categorical features.
4. **Imbalance**: churners are the minority class, so all models use class weights.
5. **Model selection**: 5-fold stratified cross-validation on the training split; the model with the highest F1 for the churn class is refitted on the full training split and saved as one pipeline.
6. **Evaluation**: one run on the test set (`evaluate.py`).

## Exploratory analysis (from this dataset)
- **Contract**: churn is 42.7% for month-to-month, 11.3% for one-year and 2.8% for two-year contracts.
- **Tenure**: churn falls from 47.4% (0-12 months) to 6.6% (over 60 months). Churned customers averaged 18.0 months of tenure versus 37.6 for those who stayed.
- **Internet service**: fiber optic 41.9%, DSL 19.0%, no internet 7.4%.
- **Payment method**: electronic check 45.3%; mailed check 19.1%, bank transfer 16.7%, credit card 15.2%.
- **Monthly charges**: churned customers paid more on average ($74.44 vs $61.27).
- **Other**: paperless billing 33.6% vs 16.3% without; customers without tech support 41.6% vs 15.2% with it.

These are associations in historical data, not proof of cause. Charts are in the dashboard and saved as `reports/eda_*.png`.

## Why recall and precision matter
Only 26.5% of customers churn, so a model that always predicts "stay" would already be about 73% accurate. Accuracy therefore hides what the business cares about:
- **Recall**: the share of real churners the model finds. Missing a churner means losing revenue with no chance to react.
- **Precision**: the share of flagged customers who really would leave. Low precision wastes retention offers on loyal customers.

F1 balances both, so it was used to choose the model.

## Results (produced by running `train.py` and `evaluate.py`)
Cross-validation on the training split (5 folds):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Hist Gradient Boosting | 0.753 | 0.523 | 0.799 | 0.632 | 0.848 |
| Random Forest | 0.774 | 0.558 | 0.726 | 0.631 | 0.845 |
| Logistic Regression | 0.749 | 0.517 | 0.801 | 0.628 | 0.846 |

The three models are very close; the selected one (Hist Gradient Boosting) leads by a tiny margin in F1.

Selected model on the held-out test set (1,409 customers, threshold 0.5):

| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0.748 | 0.516 | 0.797 | 0.627 | 0.843 |

Confusion matrix: 756 true negatives, 279 false positives, 76 false negatives, 298 true positives. Accuracy is only slightly above the 73.5% majority-class baseline; the value of the model is its recall (about 80% of churners found) at the cost of precision (about half of flagged customers actually churn). Results may differ slightly with other library versions.

The most influential features by permutation importance (drop in test ROC-AUC) are `Contract` and `tenure`, followed at a much smaller scale by `InternetService` and `OnlineSecurity`. This shows what the model uses, not what causes churn.

## Installation (Windows PowerShell, Python 3.12)
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
If activation is blocked, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first.

## Usage
```powershell
python train.py       # downloads the dataset (first run), compares models, saves models/churn_pipeline.joblib
python evaluate.py    # test metrics, confusion matrix, ROC curve, importance, EDA charts -> reports/
streamlit run app.py  # open http://localhost:8501
```
Tests (optional): `python -m pip install -r requirements-dev.txt` then `python -m pytest`.

## Project structure
```
app.py             Streamlit dashboard
train.py           Model comparison and saving
evaluate.py        Test-set evaluation and report files
preprocessing.py   Dataset download, cleaning, split, preprocessing pipeline
predict.py         Model loading, input validation, prediction, explanation
plots.py           Matplotlib figures used by the app and evaluate.py
tests/             pytest tests (synthetic data, plus real-data schema check)
data/              telco_churn.csv (downloaded automatically, not committed)
models/            churn_pipeline.joblib and model_info.json
reports/           metrics, comparison table and charts
screenshots/       add your own screenshots here
```
`predict.py` and `plots.py` were added to the basic structure so the app, evaluation and tests share one implementation instead of duplicating code.

## Troubleshooting
- **Dataset download fails**: download the CSV from the Kaggle link above and save it as `data/telco_churn.csv`.
- **"Model file not found"**: run `python train.py`.
- **Model fails to load or version warning**: the model was saved with a different scikit-learn version; run `python train.py` again.
- **`pyarrow requires NumPy 2.0`**: use the versions in `requirements.txt` (pyarrow 17.0.0 with NumPy 1.26.4).
- **Install errors**: check `python --version` shows 3.12 and that you ran `python -m pip` inside the activated `.venv`.

## Limitations
- One fictional telecom sample; results may not transfer to other businesses.
- The model finds associations, not causes. The per-customer explanation is a one-feature-at-a-time what-if analysis of the model, not a causal claim.
- Probabilities are not calibrated and the 0.5 threshold was not tuned to business costs.
- The data has no time dimension, so trends over time cannot be studied.

## Future improvements
- Tune the decision threshold using retention-offer cost and customer value.
- Hyperparameter search and probability calibration.
- SHAP values for per-customer explanations.

## Screenshots
Run the app and save these to `screenshots/`: customer prediction result, churn analysis page, model performance page.

## GitHub
```powershell
git init
git add .
git commit -m "Customer churn prediction project"
git branch -M main
git remote add origin https://github.com/<your-username>/customer-churn-prediction.git
git push -u origin main
```
`.gitignore` excludes the virtual environment, caches and `data/*.csv`. The trained model and reports are committed (the model file is small) so the deployed app can start without training.

## Deploy on Streamlit Community Cloud
1. Push the repository to GitHub (above).
2. At https://share.streamlit.io choose the repository, branch `main`, main file `app.py`; under Advanced settings select Python 3.12.
3. Deploy. On first start the app downloads the dataset itself (the CSV is not in Git) and loads `models/churn_pipeline.joblib`.
4. If the model was trained with different library versions than `requirements.txt` pins, retrain locally with the pinned versions and commit the new model.

This project has not been deployed by the author of this README; deployment status depends on you.

## Attribution
Dataset: IBM Telco Customer Churn sample data, published by IBM (https://github.com/IBM/telco-customer-churn-on-icp4d) and mirrored on Kaggle.
