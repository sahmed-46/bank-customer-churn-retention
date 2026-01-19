
## 🚀 How to Run This Project

The project is organized into three main notebooks that build on each other.

### 1. Data Prep & Feature Engineering

**Notebook:** `notebooks/01_bank_churn_eda_features.ipynb`

This notebook:

- Loads the raw banking churn CSV

- Drops ID / leakage columns:
    `CustomerId`, `Surname`, `First Name`, `Address`, `Contact Information`,
    `Date of Birth`, `Churn Reason`, `Churn Date`

- Engineers features:
    Tenure buckets (`tenure_bucket`)
    High-balance flag (`high_balance`)
    Complaints flag (`has_complaints`)
    `Occupation_freq` (frequency-encoded version of `Occupation`)

- One-hot encodes low-cardinality categoricals:
    `Gender`, `Marital Status`, `Education Level`,
    `Customer Segment`, `Preferred Communication Channel`, `tenure_bucket`

- Splits into train/validation/test

- Saves processed data to:

```text
processed_bank/train.parquet
processed_bank/val.parquet
processed_bank/test.parquet


*** Run this first, once per dataset ***

----------------------------------------------------------------------------------------------------

2. Modeling & Retention Recommender

Notebook: notebooks/02_bank_churn_modeling_recommender.ipynb

This notebook:

- Loads processed train/val/test parquet files

-Trains:
    Logistic Regression (baseline)
    XGBoost (main model)

- Evaluates using:
    ROC-AUC, PR-AUC
    Recall@K (top 5%, 10%, 20%)
    Lift curves

- Chooses XGBoost as the final model

- Computes:
    churn_score (predicted probability of churn)
    clv_est
    expected_lost_value

- Segments customers into:
    affluent_multi_product
    high_complaint_risk
    new_customer
    single_product_low_engagement
    mass_market

- Assigns recommended_action for top 20% highest-risk customers

- Saves:
    outputs_bank/bank_churn_recommendations.csv
    model file: models/xgb_bank_churn.json

*** Run this second, after Notebook 1 ***

----------------------------------------------------------------------------------------------------

3. Inference on New Customers

Notebook: notebooks/03_inference_on_new_customers.ipynb

This notebook:

- Loads the trained XGBoost model and processed schema

- Reads ~15k new customers from:
    data/new_bank_customers.csv

-Applies the same preprocessing:

Drops ID/leakage columns

- Adds engineered features (tenure_bucket, high_balance, has_complaints, Occupation_freq)

- One-hot encodes the same categorical columns

- Aligns columns with training feature set

- Predicts:
    churn_score
    clv_est
    expected_lost_value
    segment
    recommended_action

- Exports final scoring file:
    outputs_bank/new_customers_churn_predictions.csv


*** Run this third, whenever you get a new batch of customers to score ***

----------------------------------------------------------------------------------------------------

You can tweak paths/names if yours differ, but this is basically plug-and-play.

---

## 2️⃣ Extra summary stats for Notebook 3 (impact view)

Add these **at the end of Notebook 3**, after you’ve created `df_export`.

Assuming `df_export` has at least:

- `churn_score`
- `expected_lost_value`
- `recommended_action`

###############################################################################

Cell A – How many customers we recommend contacting?

```python
total_customers = len(df_export)
to_contact = (df_export["recommended_action"] != "no_action").sum()
contact_rate = to_contact / total_customers

print(f"Total new customers: {total_customers}")
print(f"Customers flagged for outreach: {to_contact} ({contact_rate:.1%})")

###############################################################################

Cell B – What share of expected lost value do those customers represent?

total_elv = df_export["expected_lost_value"].sum()
elv_targeted = df_export.loc[df_export["recommended_action"] != "no_action", "expected_lost_value"].sum()

coverage = elv_targeted / total_elv if total_elv > 0 else np.nan

print(f"Total expected lost value (all new customers): {total_elv:,.2f}")
print(f"Expected lost value within targeted group:     {elv_targeted:,.2f}")
print(f"Share of expected loss covered by actions:     {coverage:.1%}")

###############################################################################

Cell C – Breakdown of actions by segment
action_segment_pivot = (
    df_export
    .groupby(["segment", "recommended_action"])
    .size()
    .reset_index(name="count")
)

action_segment_pivot

###############################################################################

Optional: visualize:

plt.figure(figsize=(8,5))
pivot_plot = (
    df_export
    .groupby(["segment", "recommended_action"])
    .size()
    .reset_index(name="count")
)

sns.barplot(
    data=pivot_plot,
    x="segment",
    y="count",
    hue="recommended_action"
)
plt.xticks(rotation=30, ha="right")
plt.title("Recommended actions per segment (new customers)")
plt.ylabel("Number of customers")
plt.show()

###############################################################################