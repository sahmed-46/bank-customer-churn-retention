"""Score the latest customer snapshots from Supabase/PostgreSQL.

Required environment variables:
  SUPABASE_DB_URL: PostgreSQL connection string stored as a CI secret.
  MODEL_VERSION: optional model identifier, default xgb_bank_churn_v1.

The worker writes a dashboard-compatible CSV export and versioned predictions
back to the database. It intentionally does not export PII fields.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
import xgboost as xgb


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXPORT_DIR = DATA / "db_exports"
MODEL_VERSION = os.getenv("MODEL_VERSION", "xgb_bank_churn_v1")
DB_URL = os.environ.get("SUPABASE_DB_URL")
if not DB_URL:
    raise SystemExit("SUPABASE_DB_URL is required")

sys.path.insert(0, str(ROOT / "dashboard" / "scripts"))
from build_data import DROP_COLS, engineer  # noqa: E402


SNAPSHOT_COLUMNS = [
    "snapshot_id", "customer_id", "external_customer_id", "observed_at", "gender",
    "marital_status", "number_of_dependents", "occupation", "income", "education_level",
    "customer_tenure", "customer_segment", "preferred_communication_channel",
    "credit_score", "credit_history_length", "outstanding_loans", "balance",
    "num_of_products", "num_complaints",
]


def fetch_latest_snapshots(connection) -> pd.DataFrame:
    query = """
        select distinct on (s.customer_id)
            s.snapshot_id, s.customer_id, c.external_customer_id, s.observed_at,
            s.gender, s.marital_status, s.number_of_dependents, s.occupation,
            s.income, s.education_level, s.customer_tenure, s.customer_segment,
            s.preferred_communication_channel, s.credit_score,
            s.credit_history_length, s.outstanding_loans, s.balance,
            s.num_of_products, s.num_complaints
        from public.customer_snapshots s
        join public.customers c on c.customer_id = s.customer_id
        order by s.customer_id, s.observed_at desc, s.snapshot_id desc
    """
    rows = connection.execute(query).fetchall()
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def raw_frame(snapshots: pd.DataFrame) -> pd.DataFrame:
    frame = snapshots.rename(columns={
        "external_customer_id": "CustomerId",
        "gender": "Gender",
        "marital_status": "Marital Status",
        "number_of_dependents": "Number of Dependents",
        "occupation": "Occupation",
        "income": "Income",
        "education_level": "Education Level",
        "customer_tenure": "Customer Tenure",
        "customer_segment": "Customer Segment",
        "preferred_communication_channel": "Preferred Communication Channel",
        "credit_score": "Credit Score",
        "credit_history_length": "Credit History Length",
        "outstanding_loans": "Outstanding Loans",
        "balance": "Balance",
        "num_of_products": "NumOfProducts",
        "num_complaints": "NumComplaints",
    })
    numeric_columns = [
        "Number of Dependents", "Income", "Customer Tenure", "Credit Score",
        "Credit History Length", "Outstanding Loans", "Balance", "NumOfProducts", "NumComplaints",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def main() -> None:
    train = pd.read_csv(DATA / "bank_customer_churn_copy.csv").drop(columns=DROP_COLS)
    train_x = train.drop(columns=["Churn Flag"])
    train_x, _, _, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        train_x, train["Churn Flag"], test_size=0.2, stratify=train["Churn Flag"], random_state=42
    )
    train_x, _, _, _ = __import__("sklearn.model_selection", fromlist=["train_test_split"]).train_test_split(
        train_x, train.loc[train_x.index, "Churn Flag"], test_size=0.25,
        stratify=train.loc[train_x.index, "Churn Flag"], random_state=42
    )
    occupation_counts = train_x["Occupation"].value_counts()

    model = xgb.XGBClassifier()
    model.load_model(DATA / "models" / "xgb_bank_churn.json")
    feature_names = model.get_booster().feature_names

    with psycopg.connect(DB_URL) as connection:
        cursor = connection.cursor()
        snapshots = fetch_latest_snapshots(connection)
        if snapshots.empty:
            raise SystemExit("No customer snapshots found")
        source_watermark = snapshots["observed_at"].max()
        run = connection.execute(
            "insert into public.scoring_runs (model_name, model_version, source_watermark, rows_read) "
            "values (%s, %s, %s, %s) returning run_id",
            ("xgboost", MODEL_VERSION, source_watermark, len(snapshots)),
        ).fetchone()
        run_id = run[0]

        raw = raw_frame(snapshots)
        features = engineer(raw.drop(columns=["CustomerId"]), train_x, occupation_counts, feature_names)
        scores = model.predict_proba(features)[:, 1]
        balance = raw["Balance"].fillna(train_x["Balance"].median())
        loans = raw["Outstanding Loans"].fillna(train_x["Outstanding Loans"].median())
        products = raw["NumOfProducts"].fillna(0)
        clv = balance + 0.5 * loans + 200 * products
        balance_median = train_x["Balance"].median()
        segments = np.select(
            [
                (balance > balance_median) & (products >= 3),
                raw["NumComplaints"].fillna(0) > 0,
                raw["Customer Tenure"].fillna(0) < 6,
                products == 1,
            ],
            ["affluent_multi_product", "high_complaint_risk", "new_customer", "single_product_low_engagement"],
            default="mass_market",
        )
        cutoff = np.quantile(scores, 0.8)
        actions = np.where(scores < cutoff, "no_action", "general_retention_email")
        actions[(scores >= cutoff) & (segments == "affluent_multi_product")] = "personal_banker_outreach_rate_offer"
        actions[(scores >= cutoff) & (segments == "high_complaint_risk")] = "service_recovery_and_fee_refund"
        actions[(scores >= cutoff) & (segments == "new_customer")] = "onboarding_support_and_welcome_offer"
        actions[(scores >= cutoff) & (segments == "single_product_low_engagement")] = "cross_sell_bundle_offer"

        cursor.executemany(
            "insert into public.customer_predictions "
            "(customer_id, snapshot_id, run_id, churn_score, clv_est, expected_lost_value, segment, recommended_action, model_version) "
            "values (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "on conflict (customer_id, snapshot_id, model_version) do update set "
            "run_id = excluded.run_id, churn_score = excluded.churn_score, clv_est = excluded.clv_est, "
            "expected_lost_value = excluded.expected_lost_value, segment = excluded.segment, "
            "recommended_action = excluded.recommended_action, predicted_at = now()",
            [
                (row.customer_id, row.snapshot_id, run_id, float(score), float(value), float(score * value), segment, action, MODEL_VERSION)
                for row, score, value, segment, action in zip(snapshots.itertuples(), scores, clv, segments, actions)
            ],
        )
        connection.execute(
            "update public.scoring_runs set completed_at = now(), rows_scored = %s, status = 'completed' where run_id = %s",
            (len(snapshots), run_id),
        )

        dashboard = raw[["CustomerId", "Customer Tenure", "Customer Segment", "Balance", "NumOfProducts", "NumComplaints"]].copy()
        dashboard["churn_score"] = scores
        dashboard["clv_est"] = clv
        dashboard["expected_lost_value"] = scores * clv
        dashboard["segment"] = segments
        dashboard["recommended_action"] = actions
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        raw.to_csv(EXPORT_DIR / "new_customer_data.csv", index=False)
        dashboard.to_csv(EXPORT_DIR / "dashboard_inference_predictions.csv", index=False)
        print(f"Scored {len(snapshots)} customers in run {run_id}")


if __name__ == "__main__":
    main()
