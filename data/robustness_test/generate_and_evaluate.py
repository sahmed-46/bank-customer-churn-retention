"""Generate a realistic labeled banking cohort and evaluate the saved model.

This is a reproducible robustness test. It intentionally includes skewed fields,
outliers, missing values, class imbalance, and moderate population/concept drift
while preserving the raw column contract used by the project notebooks.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = DATA / "robustness_test"
SEED = 20261123
N_ROWS = 10_000
TARGET = "Churn Flag"
DROP_COLS = [
    "CustomerId", "Surname", "First Name", "Address",
    "Contact Information", "Date of Birth", "Churn Reason", "Churn Date",
]
OHE_COLS = [
    "Gender", "Marital Status", "Education Level",
    "Customer Segment", "Preferred Communication Channel", "tenure_bucket",
]


def _clip(series, low, high):
    return np.clip(series, low, high)


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    n = N_ROWS

    # Skewed and heavy-tailed financial variables.
    income = rng.lognormal(mean=np.log(62_000), sigma=0.72, size=n)
    balance = rng.gamma(shape=1.55, scale=72_000, size=n)
    balance_tail = rng.random(n) < 0.07
    balance[balance_tail] += rng.lognormal(np.log(120_000), 0.85, balance_tail.sum())
    loans = rng.gamma(shape=1.25, scale=38_000, size=n)
    products = rng.choice([1, 2, 3, 4], size=n, p=[0.57, 0.28, 0.12, 0.03])
    tenure = _clip(rng.exponential(scale=7.5, size=n), 0.2, 30).round().astype(int)
    dependents = _clip(rng.poisson(1.15, size=n), 0, 7).astype(int)
    complaints = rng.poisson(0.28, size=n).astype(int)
    credit_score = _clip(rng.normal(690, 74, size=n), 300, 850)
    credit_history = _clip(tenure * 1.3 + rng.normal(4.5, 5.5, size=n), 0.5, 42)

    # Real-life data quality issues: missingness and a few extreme outliers.
    income[rng.random(n) < 0.025] = np.nan
    balance[rng.random(n) < 0.018] = np.nan
    loans[rng.random(n) < 0.02] = np.nan
    credit_score[rng.random(n) < 0.012] = np.nan
    outlier_rows = rng.choice(n, size=round(n * 0.008), replace=False)
    income[outlier_rows] *= rng.uniform(3.5, 9.0, len(outlier_rows))
    balance[outlier_rows] *= rng.uniform(2.0, 5.0, len(outlier_rows))

    gender = rng.choice(["Male", "Female"], size=n, p=[0.49, 0.51]).astype(object)
    marital = rng.choice(["Single", "Married", "Divorced", "Widowed"], size=n, p=[0.35, 0.48, 0.12, 0.05]).astype(object)
    education = rng.choice(["High School", "Diploma", "Bachelor's", "Master's", "PhD"], size=n, p=[0.22, 0.23, 0.34, 0.17, 0.04]).astype(object)
    occupation = rng.choice(["Salaried", "Self-employed", "Business Owner", "Professional", "Retired", "Student"], size=n, p=[0.45, 0.18, 0.10, 0.16, 0.08, 0.03]).astype(object)
    customer_segment = rng.choice(["Retail", "SME", "Corporate"], size=n, p=[0.72, 0.22, 0.06]).astype(object)
    channel = rng.choice(["Email", "Phone", "Branch", "App"], size=n, p=[0.39, 0.27, 0.18, 0.16]).astype(object)
    for values, rate in [(gender, 0.012), (marital, 0.018), (education, 0.015), (occupation, 0.02), (channel, 0.012)]:
        values[rng.random(n) < rate] = None

    # Labels are deliberately not identical to the training-generating process:
    # additional digital-engagement and complaint effects create mild drift.
    b = np.nan_to_num(balance, nan=np.nanmedian(balance))
    inc = np.nan_to_num(income, nan=np.nanmedian(income))
    loan_amount = np.nan_to_num(loans, nan=np.nanmedian(loans))
    cs = np.nan_to_num(credit_score, nan=690)
    logit = (
        -4.4
        + 0.000007 * b
        + 0.0000014 * loan_amount
        - 0.0062 * (cs - 650)
        - 0.055 * tenure
        + 0.52 * (products == 1)
        + 0.78 * (complaints > 0)
        + 0.35 * (channel == "Branch")
        + 0.28 * (channel == "App")
        + 0.16 * (inc < 35_000)
        + 0.25 * (customer_segment == "SME")
        + rng.normal(0, 0.62, n)
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_probability).astype(int)

    first_names = np.array(["Ava", "Noah", "Mia", "Liam", "Emma", "Ethan", "Sara", "Omar"])
    last_names = np.array(["Patel", "Smith", "Garcia", "Kim", "Brown", "Wilson", "Khan", "Lee"])
    df = pd.DataFrame({
        "CustomerId": [f"SYN-{i:07d}" for i in range(1, n + 1)],
        "Surname": rng.choice(last_names, n),
        "First Name": rng.choice(first_names, n),
        "Date of Birth": pd.to_datetime("1940-01-01") + pd.to_timedelta(rng.integers(0, 23_000, n), unit="D"),
        "Gender": gender,
        "Marital Status": marital,
        "Number of Dependents": dependents,
        "Occupation": occupation,
        "Income": income.round(2),
        "Education Level": education,
        "Address": [f"{rng.integers(1, 9999)} Sample Avenue" for _ in range(n)],
        "Contact Information": [f"+1-555-{rng.integers(1000000, 9999999)}" for _ in range(n)],
        "Customer Tenure": tenure,
        "Customer Segment": customer_segment,
        "Preferred Communication Channel": channel,
        "Credit Score": credit_score.round(1),
        "Credit History Length": credit_history.round(1),
        "Outstanding Loans": loans.round(2),
        TARGET: churn,
        "Churn Reason": np.where(churn == 1, rng.choice(["Service", "Fees", "Competitor", "Unknown"], n), None),
        "Churn Date": np.where(churn == 1, "2026-10-31", None),
        "Balance": balance.round(2),
        "NumOfProducts": products,
        "NumComplaints": complaints,
    })
    return df


def engineer(frame: pd.DataFrame, balance_reference: pd.DataFrame, occupation_counts: pd.Series, feature_names: list[str]) -> pd.DataFrame:
    df = frame.copy()
    df["tenure_bucket"] = pd.cut(
        df["Customer Tenure"], bins=[0, 6, 12, 36, 100], labels=["0-6m", "6-12m", "1-3y", "3y+"]
    )
    df["high_balance"] = (df["Balance"] > balance_reference["Balance"].median()).astype(int)
    df["has_complaints"] = (df["NumComplaints"] > 0).astype(int)
    df["Occupation_freq"] = df["Occupation"].map(occupation_counts).fillna(0)
    df = df.drop(columns=["Occupation"])
    encoded = pd.get_dummies(df, columns=OHE_COLS, drop_first=True)
    return encoded.reindex(columns=feature_names, fill_value=0).astype(float)


def recall_at_k(y_true, scores, fraction):
    cutoff = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-scores)[:cutoff]
    return float(np.asarray(y_true)[order].sum() / max(np.asarray(y_true).sum(), 1))


def metrics(y_true, scores):
    predicted = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "records": int(len(y_true)),
        "churn_rate": float(np.mean(y_true)),
        "rmse_probability": float(mean_squared_error(y_true, scores) ** 0.5),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, predicted)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "recall_at_5pct": recall_at_k(y_true, scores, 0.05),
        "recall_at_10pct": recall_at_k(y_true, scores, 0.10),
        "recall_at_20pct": recall_at_k(y_true, scores, 0.20),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generated = generate_dataset()
    generated.to_csv(OUT / "synthetic_banking_stress_test.csv", index=False)

    model = xgb.XGBClassifier()
    model.load_model(DATA / "models" / "xgb_bank_churn.json")
    feature_names = model.get_booster().feature_names

    train_raw = pd.read_csv(DATA / "bank_customer_churn_copy.csv").drop(columns=DROP_COLS)
    train_x, train_test_x, train_y, train_test_y = train_test_split(
        train_raw.drop(columns=[TARGET]), train_raw[TARGET], test_size=0.2, stratify=train_raw[TARGET], random_state=42
    )
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.25, stratify=train_y, random_state=42
    )
    occupation_counts = train_x["Occupation"].value_counts()
    balance_reference = train_x

    train_features = engineer(train_x, balance_reference, occupation_counts, feature_names)
    val_features = engineer(val_x, balance_reference, occupation_counts, feature_names)
    test_features = engineer(train_test_x, balance_reference, occupation_counts, feature_names)
    new_raw = generated.drop(columns=DROP_COLS)
    new_features = engineer(new_raw.drop(columns=[TARGET]), balance_reference, occupation_counts, feature_names)

    scores = {
        "Training": (train_y.to_numpy(), model.predict_proba(train_features)[:, 1]),
        "Validation": (val_y.to_numpy(), model.predict_proba(val_features)[:, 1]),
        "Test": (train_test_y.to_numpy(), model.predict_proba(test_features)[:, 1]),
        "New synthetic stress test": (generated[TARGET].to_numpy(), model.predict_proba(new_features)[:, 1]),
    }
    results = pd.DataFrame([{ "dataset": name, **metrics(y, s) } for name, (y, s) in scores.items()])
    results.to_csv(OUT / "model_comparison_results.csv", index=False)

    new_predictions = generated[["CustomerId", TARGET]].copy()
    new_scores = scores["New synthetic stress test"][1]
    new_predictions["churn_probability"] = new_scores
    new_predictions["predicted_churn"] = (new_predictions["churn_probability"] >= 0.5).astype(int)
    new_predictions.to_csv(OUT / "synthetic_stress_test_predictions.csv", index=False)

    # Dashboard-compatible inference export. Numeric display fields are median-filled
    # only for the UI so missing raw values do not disappear from filters or expected-loss totals.
    dashboard_predictions = generated[["CustomerId", "Customer Tenure", "Customer Segment", "Balance", "NumOfProducts", "NumComplaints"]].copy()
    dashboard_predictions["Balance"] = dashboard_predictions["Balance"].fillna(dashboard_predictions["Balance"].median())
    dashboard_predictions["churn_score"] = new_scores
    dashboard_predictions["clv_est"] = (
        dashboard_predictions["Balance"]
        + 0.5 * generated["Outstanding Loans"].fillna(generated["Outstanding Loans"].median())
        + 200 * dashboard_predictions["NumOfProducts"]
    )
    dashboard_predictions["expected_lost_value"] = dashboard_predictions["churn_score"] * dashboard_predictions["clv_est"]
    balance_median = balance_reference["Balance"].median()
    dashboard_predictions["segment"] = np.select(
        [
            (dashboard_predictions["Balance"] > balance_median) & (dashboard_predictions["NumOfProducts"] >= 3),
            dashboard_predictions["NumComplaints"] > 0,
            dashboard_predictions["Customer Tenure"] < 6,
            dashboard_predictions["NumOfProducts"] == 1,
        ],
        ["affluent_multi_product", "high_complaint_risk", "new_customer", "single_product_low_engagement"],
        default="mass_market",
    )
    action_cutoff = np.quantile(new_scores, 0.8)
    dashboard_predictions["recommended_action"] = "no_action"
    targeted = dashboard_predictions["churn_score"] >= action_cutoff
    dashboard_predictions.loc[targeted & (dashboard_predictions["segment"] == "affluent_multi_product"), "recommended_action"] = "personal_banker_outreach_rate_offer"
    dashboard_predictions.loc[targeted & (dashboard_predictions["segment"] == "high_complaint_risk"), "recommended_action"] = "service_recovery_and_fee_refund"
    dashboard_predictions.loc[targeted & (dashboard_predictions["segment"] == "new_customer"), "recommended_action"] = "onboarding_support_and_welcome_offer"
    dashboard_predictions.loc[targeted & (dashboard_predictions["segment"] == "single_product_low_engagement"), "recommended_action"] = "cross_sell_bundle_offer"
    dashboard_predictions.loc[targeted & (dashboard_predictions["segment"] == "mass_market"), "recommended_action"] = "general_retention_email"
    dashboard_predictions.to_csv(OUT / "dashboard_inference_predictions.csv", index=False)

    gaps = results.set_index("dataset")
    recommendations = []
    new = gaps.loc["New synthetic stress test"]
    test = gaps.loc["Test"]
    if new.roc_auc < test.roc_auc - 0.03:
        recommendations.append("Investigate population/concept drift; refresh training data with recent and stress-test-like customers.")
    if new.rmse_probability > test.rmse_probability + 0.03:
        recommendations.append("Improve probability calibration using Platt scaling or isotonic calibration on a recent holdout set.")
    if new.recall_at_20pct < 0.90:
        recommendations.append("Re-tune ranking and threshold policy; top-20% outreach is missing too many churners in the new cohort.")
    if new.recall < 0.70:
        recommendations.append("Review the 0.50 classification threshold and consider a cost-sensitive threshold based on retention capacity.")
    if not recommendations:
        recommendations.append("Monitor drift and recalibrate periodically; no major metric-based intervention was triggered by this stress test.")
    pd.DataFrame({"improvement_recommendation": recommendations}).to_csv(OUT / "improvement_recommendations.csv", index=False)

    metadata = {
        "seed": SEED,
        "records": N_ROWS,
        "synthetic_features": ["skewed income/balance/loans", "heavy-tail outliers", "missing values", "class imbalance", "population drift", "concept drift", "categorical variation"],
        "model": "Saved XGBoost model from data/models/xgb_bank_churn.json",
        "threshold": 0.5,
        "note": "RMSE is computed on predicted churn probabilities; it is the square root of the Brier score.",
    }
    (OUT / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print("Saved stress-test artifacts to", OUT)


if __name__ == "__main__":
    main()
