"""Build compact dashboard data from the notebook outputs.

Reads the scored CSVs in ../data/outputs_bank, rebuilds the model features from the
raw CSVs exactly as notebooks 01-03 do, and writes to public/data:

  manifest.json            schema, category dictionaries, feature labels, global drivers
  <dataset>.bin            row-level columns (float32 + uint8), little-endian
  <dataset>-drivers.bin    top SHAP contributions per row, loaded on demand

Run from the dashboard directory:  python scripts/build_data.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
OUT = Path(__file__).resolve().parents[1] / "public" / "data"

TARGET = "Churn Flag"
DROP_COLS = [
    "CustomerId", "Surname", "First Name", "Address",
    "Contact Information", "Date of Birth", "Churn Reason", "Churn Date",
]
OHE_COLS = [
    "Gender", "Marital Status", "Education Level",
    "Customer Segment", "Preferred Communication Channel", "tenure_bucket",
]
TOP_DRIVERS = 3
MAX_SCORE_DRIFT = 1e-6

FEATURE_LABELS = {
    "Number of Dependents": "Dependents",
    "Income": "Income",
    "Customer Tenure": "Tenure (years)",
    "Credit Score": "Credit score",
    "Credit History Length": "Credit history length",
    "Outstanding Loans": "Outstanding loans",
    "Balance": "Balance",
    "NumOfProducts": "Products held",
    "NumComplaints": "Complaints",
    "high_balance": "Above-median balance",
    "has_complaints": "Has complaints",
    "Occupation_freq": "Occupation frequency",
    "Gender_Male": "Gender: male",
    "Marital Status_Married": "Married",
    "Marital Status_Single": "Single",
    "Education Level_Diploma": "Education: diploma",
    "Education Level_High School": "Education: high school",
    "Education Level_Master's": "Education: master's",
    "Customer Segment_Retail": "Segment: retail",
    "Customer Segment_SME": "Segment: SME",
    "Preferred Communication Channel_Phone": "Prefers phone",
    "tenure_bucket_6-12m": "Tenure bucket 6-12",
    "tenure_bucket_1-3y": "Tenure bucket 12-36",
    "tenure_bucket_3y+": "Tenure bucket 36+",
}
BINARY_FEATURES = {name for name in FEATURE_LABELS if "_" in name and name != "Occupation_freq"}


def engineer(frame, balance_reference, occupation_counts, feature_names):
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


def labeled_features(feature_names):
    raw = pd.read_csv(DATA / "bank_customer_churn_copy.csv").drop(columns=DROP_COLS)
    X, y = raw.drop(columns=[TARGET]), raw[TARGET]
    X_full, X_test, y_full, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_full, y_full, test_size=0.25, stratify=y_full, random_state=42
    )
    occupation_counts = X_train["Occupation"].value_counts()
    parts = [engineer(part, X_train, occupation_counts, feature_names) for part in (X_train, X_val, X_test)]
    raw_rows = pd.concat([X_train, X_val, X_test])
    labels = np.concatenate([y_train, y_val, y_test])
    return pd.concat(parts), raw_rows, labels, (len(X_train), len(X_val), len(X_test))


def inference_features(feature_names, raw_path):
    train_raw = pd.read_csv(DATA / "bank_customer_churn_copy.csv").drop(columns=DROP_COLS)
    train_x, _, _, _ = train_test_split(
        train_raw.drop(columns=[TARGET]), train_raw[TARGET], test_size=0.2, stratify=train_raw[TARGET], random_state=42
    )
    train_x, _, _, _ = train_test_split(
        train_x, train_raw.loc[train_x.index, TARGET], test_size=0.25, stratify=train_raw.loc[train_x.index, TARGET], random_state=42
    )
    new_raw = pd.read_csv(raw_path).drop(columns=DROP_COLS, errors="ignore")
    features = engineer(new_raw, train_x, train_x["Occupation"].value_counts(), feature_names)
    return features, new_raw


def explain(booster, features, scores, name):
    matrix = xgb.DMatrix(features.values, feature_names=list(features.columns))
    margin = booster.predict(matrix, output_margin=True)
    drift = np.abs(1 / (1 + np.exp(-margin)) - scores).max()
    if drift > MAX_SCORE_DRIFT:
        raise SystemExit(f"{name}: rebuilt features do not reproduce saved scores (max drift {drift:.2e})")
    contributions = booster.predict(matrix, pred_contribs=True)
    return contributions[:, :-1], float(contributions[0, -1]), drift


def write_columns(path, columns):
    """Write named numpy arrays back to back, 4-byte aligned. Returns the column layout."""
    layout, offset = [], 0
    with path.open("wb") as handle:
        for name, array in columns:
            padding = (-offset) % 4
            handle.write(b"\0" * padding)
            offset += padding
            data = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<"), copy=False)
            handle.write(data.tobytes())
            layout.append({"name": name, "type": data.dtype.name, "offset": offset, "length": int(data.size)})
            offset += data.nbytes
    return layout


def build_dataset(name, scored, raw_rows, labels, contributions, features, feature_names, dictionaries):
    def codes(values, key):
        lookup = {value: index for index, value in enumerate(dictionaries[key])}
        return np.array([lookup[value] for value in values], dtype=np.uint8)

    churned = np.full(len(scored), 255, dtype=np.uint8) if labels is None else labels.astype(np.uint8)
    core = write_columns(OUT / f"{name}.bin", [
        ("churnScore", scored["churn_score"].to_numpy(np.float32)),
        ("clv", scored["clv_est"].to_numpy(np.float32)),
        ("expectedLoss", scored["expected_lost_value"].to_numpy(np.float32)),
        ("balance", scored["Balance"].to_numpy(np.float32)),
        ("tenure", scored["Customer Tenure"].to_numpy(np.uint8)),
        ("products", scored["NumOfProducts"].to_numpy(np.uint8)),
        ("churned", churned),
        ("segment", codes(scored["segment"], "segment")),
        ("action", codes(scored["recommended_action"], "action")),
        ("customerSegment", codes(raw_rows["Customer Segment"], "customerSegment")),
    ])

    order = np.argsort(-np.abs(contributions), axis=1)[:, :TOP_DRIVERS]
    rows = np.arange(len(scored))[:, None]
    drivers = write_columns(OUT / f"{name}-drivers.bin", [
        ("contribution", contributions[rows, order].astype(np.float32).ravel()),
        ("value", features.to_numpy()[rows, order].astype(np.float32).ravel()),
        ("feature", order.astype(np.uint8).ravel()),
    ])
    return {"rows": len(scored), "columns": core, "drivers": drivers}


def importance(contributions, feature_names, top=10):
    mean_abs = np.abs(contributions).mean(axis=0)
    raises = (contributions > 0).mean(axis=0)
    order = np.argsort(-mean_abs)[:top]
    return [
        {"feature": int(i), "meanAbs": round(float(mean_abs[i]), 5), "raisesShare": round(float(raises[i]), 4)}
        for i in order
    ]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = xgb.XGBClassifier()
    model.load_model(DATA / "models" / "xgb_bank_churn.json")
    booster = model.get_booster()
    feature_names = booster.feature_names

    recs = pd.read_csv(DATA / "outputs_bank" / "bank_churn_recommendations.csv")
    inference_raw_path = Path(os.environ.get("DASHBOARD_INFERENCE_RAW", str(DATA / "new_customer_data.csv")))
    inference_scored_path = Path(os.environ.get("DASHBOARD_INFERENCE_SCORED", str(DATA / "outputs_bank" / "new_customers_churn_predictions.csv")))
    new_scored = pd.read_csv(inference_scored_path)

    lab_features, lab_raw, labels, sizes = labeled_features(feature_names)
    if not (labels == recs[TARGET].to_numpy()).all():
        raise SystemExit("Labeled split order no longer matches bank_churn_recommendations.csv")
    new_features, new_raw = inference_features(feature_names, inference_raw_path)

    lab_contrib, base_value, lab_drift = explain(booster, lab_features, recs["churn_score"].to_numpy(), "labeled")
    new_contrib, _, new_drift = explain(booster, new_features, new_scored["churn_score"].to_numpy(), "inference")

    dictionaries = {
        "segment": sorted(set(recs["segment"]) | set(new_scored["segment"])),
        "action": sorted(set(recs["recommended_action"]) | set(new_scored["recommended_action"])),
        "customerSegment": sorted(set(lab_raw["Customer Segment"]) | set(new_raw["Customer Segment"])),
    }

    train_end, val_end = sizes[0], sizes[0] + sizes[1]
    partitions = {
        "train": (0, train_end),
        "validation": (train_end, val_end),
        "test": (val_end, len(recs)),
    }

    manifest = {
        "featureNames": feature_names,
        "featureLabels": [FEATURE_LABELS.get(name, name) for name in feature_names],
        "binaryFeatures": [name in BINARY_FEATURES for name in feature_names],
        "baseValue": round(base_value, 6),
        "dictionaries": dictionaries,
        "partitions": {key: {"start": start, "end": end} for key, (start, end) in partitions.items()},
        "importance": {
            **{key: importance(lab_contrib[start:end], feature_names) for key, (start, end) in partitions.items()},
            "inference": importance(new_contrib, feature_names),
        },
        "datasets": {
            "labeled": {
                "source": "bank_churn_recommendations.csv",
                **build_dataset("labeled", recs, lab_raw, labels, lab_contrib, lab_features, feature_names, dictionaries),
            },
            "inference": {
                "source": inference_scored_path.name,
                **build_dataset("inference", new_scored, new_raw, None, new_contrib, new_features, feature_names, dictionaries),
            },
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))

    for path in sorted(OUT.iterdir()):
        print(f"{path.name:28s} {path.stat().st_size / 1024:9.1f} KB")
    print(f"score reproduction drift: labeled {lab_drift:.1e}, inference {new_drift:.1e}")


if __name__ == "__main__":
    main()
