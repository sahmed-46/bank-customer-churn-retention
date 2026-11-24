# Bank Customer Churn Prediction & Retention Recommender

## Overview
Built an end-to-end churn prediction and retention prioritization system to help a bank proactively retain high-risk customers while minimizing outreach costs.

## Key Results
- Trained Logistic Regression and XGBoost models on 100K customers
- Achieved AUC ≈ 0.999 and Recall@Top-20% = 100%
- Targeted top 20% of customers capturing ~99.4% of expected revenue loss
- Deployed inference pipeline scoring 15.6K new customers with segment-specific retention actions

## Tech Stack
- Python, Pandas, NumPy, Scikit-learn, XGBoost
- SQL-style data processing
- Jupyter Notebooks

## Project Structure
- `01_eda_feature_engineering.ipynb`: Data cleaning and feature engineering
- `02_modeling_recommender.ipynb`: Model training, evaluation, CLV, and recommendations
- `03_inference_new_customers.ipynb`: Scoring new customers and generating actions

## Notes
- Dataset is synthetic / anonymized
- Focus is on ranking, prioritization, and business impact

## Database refresh integration

The project includes a Supabase/PostgreSQL schema in `supabase/schema.sql`, a
database scoring worker in `jobs/score_customers.py`, and a scheduled GitHub
Actions workflow in `.github/workflows/refresh-and-deploy.yml`.

To activate the scheduled refresh:

1. Create a Supabase project and run `supabase/schema.sql`.
2. Add customer rows to `customers` and time-versioned model inputs to `customer_snapshots`.
3. Add a GitHub Actions repository secret named `SUPABASE_DB_URL` containing the PostgreSQL connection string.
4. Run the `Refresh model scores and deploy dashboard` workflow manually once.

For the included synthetic test cohort, the initial load can be reproduced with:

```powershell
python jobs/seed_csv.py data/robustness_test/synthetic_banking_stress_test.csv
python jobs/score_customers.py
```

The worker scores the latest snapshot per customer, writes versioned predictions
to PostgreSQL, rebuilds the dashboard data files, and deploys the refreshed
static dashboard. The browser never receives database credentials.
