# Bank Customer Churn Retention Project

## Project Summary

This project is an end-to-end bank customer churn prediction and retention prioritization system. It transforms customer data into churn probabilities, customer-value estimates, expected financial loss, customer segments, and recommended retention actions.

The project now includes:

- Data cleaning and feature engineering notebooks.
- Logistic Regression and XGBoost churn models.
- New-customer inference and retention recommendations.
- A React dashboard for portfolio analysis and customer prioritization.
- Model explainability using XGBoost contribution values.
- A realistic 10,000-record synthetic stress-test dataset.
- A Supabase/PostgreSQL database schema and scoring worker.
- A scheduled GitHub Actions refresh-and-deploy workflow.

The model and datasets are synthetic or anonymized and should not be used for real banking decisions without additional validation, governance, security controls, and approved production data.

## 1. Requirements

### Software requirements

- Python 3.12 or a compatible Python environment.
- Node.js 22 or compatible Node.js version.
- npm.
- Jupyter Notebook.
- PostgreSQL-compatible database for the automated database workflow.
- Git and GitHub for version control and deployment.

### Python packages

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `matplotlib`
- `seaborn`
- `pyarrow` for Parquet files when running the notebooks.
- `psycopg[binary]` for the Supabase/PostgreSQL scoring worker.

### Frontend packages

- React.
- React DOM.
- Vite.
- Recharts.
- Lucide React.
- Oxlint.

### Data requirements

The modeling workflow expects banking customer fields such as:

- Customer tenure.
- Income and balance.
- Credit score and credit history length.
- Outstanding loans.
- Number of products.
- Number of complaints.
- Occupation.
- Customer segment.
- Preferred communication channel.
- A binary churn label for supervised training and evaluation.

## 2. Architecture

```text
Raw customer data
        |
        v
Data preparation and feature engineering
        |
        v
Train / validation / test datasets
        |
        v
Logistic Regression baseline + XGBoost model
        |
        v
Churn probability, CLV, expected lost value
        |
        v
Customer segments and retention actions
        |
        +----------------------+
        |                      |
        v                      v
CSV inference exports     Supabase/PostgreSQL
        |                      |
        +----------+-----------+
                   v
        Compact dashboard data files
                   |
                   v
             React dashboard
                   |
                   v
             GitHub Pages
```

### Current static dashboard flow

The dashboard currently reads compact binary files from `dashboard/public/data/`. This keeps the browser fast and avoids exposing database credentials.

### Automated database flow

The Supabase integration uses:

1. `supabase/schema.sql` to create the database tables and dashboard view.
2. `jobs/seed_csv.py` to load an initial CSV dataset.
3. `jobs/score_customers.py` to read the latest customer snapshots, run the model, and write predictions.
4. `dashboard/scripts/build_data.py` to build compact dashboard files.
5. `.github/workflows/refresh-and-deploy.yml` to refresh and deploy the dashboard on a schedule.

The scheduled workflow is configured to run hourly and can also be started manually from GitHub Actions.

## 3. Problem Statement

Banks may know which customers have churned only after the event has occurred. By that point, the opportunity to retain the customer may already be lost.

The problem is to identify customers who are most likely to churn, prioritize them according to financial impact, and recommend an appropriate retention action. Contacting every customer is inefficient and expensive, so the system focuses retention effort on the customers with the highest combined churn risk and expected value loss.

## 4. Goal to Be Achieved

The main goal is to provide a repeatable decision-support workflow that:

- Predicts the probability that a customer will churn.
- Ranks customers by expected financial loss.
- Identifies meaningful customer groups.
- Recommends segment-specific retention actions.
- Scores new customer batches using the same preprocessing and model logic.
- Makes the results understandable through an interactive dashboard.
- Tracks model runs and predictions over time when connected to Supabase.

## 5. Technologies Used and Their Purpose

| Technology | Purpose |
|---|---|
| Python | Data preparation, feature engineering, inference, evaluation, and database scoring jobs. |
| Jupyter Notebook | Interactive analysis and reproducible modeling workflow. |
| Pandas | Tabular data loading, cleaning, transformations, grouping, and exports. |
| NumPy | Numerical operations, probability calculations, and array processing. |
| Scikit-learn | Train/test splitting, Logistic Regression, metrics, class weighting, and evaluation utilities. |
| XGBoost | Final non-linear churn prediction model. |
| Matplotlib and Seaborn | Exploratory analysis and model evaluation plots. |
| Parquet | Efficient storage of prepared train, validation, and test data. |
| React | Interactive dashboard user interface. |
| Vite | Frontend development server and production bundling. |
| Recharts | Dashboard charts and visual metric displays. |
| Lucide React | Dashboard icons. |
| JavaScript `ArrayBuffer` | Efficient loading of compact binary dashboard data. |
| Supabase PostgreSQL | Persistent storage for customers, snapshots, scoring runs, and predictions. |
| Psycopg | Python connection to PostgreSQL. |
| GitHub Actions | Scheduled scoring, dashboard rebuilding, testing, and deployment. |
| GitHub Pages | Static dashboard hosting. |

## 6. Data Procedures and Results

### Data cleaning

The workflow removes columns that are identifiers, personal information, or leakage fields:

- Customer ID.
- Customer name.
- Address.
- Contact information.
- Date of birth.
- Churn reason.
- Churn date.

The goal is to avoid privacy exposure and prevent the model from using information that would only be known after churn.

### Feature engineering

The following features are created:

- `tenure_bucket`: tenure groups such as 0-6 months, 6-12 months, 1-3 years, and 3+ years.
- `high_balance`: whether the customer balance is above the training reference median.
- `has_complaints`: whether the customer has at least one complaint.
- `Occupation_freq`: frequency encoding of occupation based on the training data.
- One-hot encoded gender, marital status, education, customer segment, communication channel, and tenure bucket.

The new-customer pipeline aligns its columns to the saved XGBoost feature schema before prediction.

### Dataset splits

The original labeled modeling dataset is split into:

- Training: 60,000 rows.
- Validation: 20,000 rows.
- Test: 20,000 rows.

The split is stratified and uses a fixed random state for reproducibility.

### Models

#### Logistic Regression baseline

Logistic Regression is used as an interpretable baseline with class weighting to address imbalance.

#### XGBoost final model

The selected model uses:

- 300 estimators.
- Maximum tree depth of 5.
- Learning rate of 0.05.
- Row subsampling of 0.8.
- Feature subsampling of 0.8.
- Binary logistic objective.
- Class-imbalance weighting.

### Original evaluation results

The saved XGBoost model achieved the following results:

| Dataset | ROC-AUC | PR-AUC | Accuracy | Recall | F1 | Recall@Top-20% |
|---|---:|---:|---:|---:|---:|---:|
| Training | 1.0000 | 1.0000 | 98.8% | 100.0% | 95.5% | 100.0% |
| Validation | 0.9997 | 0.9978 | 98.3% | 99.9% | 93.3% | 100.0% |
| Test | 0.9998 | 0.9987 | 98.4% | 100.0% | 93.9% | 100.0% |

The results are unusually strong, which is a reason to validate the model on more realistic data and monitor for leakage or synthetic-data artifacts.

### Synthetic robustness stress test

A separate 10,000-record labeled dataset was created with:

- Skewed income, balance, and loan distributions.
- Heavy-tail financial outliers.
- Missing numeric and categorical values.
- Rare product categories.
- Zero-inflated complaint counts.
- A lower churn prevalence of approximately 6.4%.
- Population distribution shift.
- Concept drift in the churn-generating process.

The existing model performed substantially worse on this stress test:

| Metric | Original Test | Synthetic Stress Test |
|---|---:|---:|
| ROC-AUC | 0.9998 | 0.4307 |
| PR-AUC | 0.9987 | 0.0517 |
| Probability RMSE | 0.1013 | 0.2920 |
| Accuracy | 98.4% | 90.7% |
| Precision | 88.6% | 2.6% |
| Recall | 100.0% | 1.2% |
| F1 score | 93.9% | 1.7% |
| Recall@Top-20% | 100.0% | 9.3% |

The stress test shows that accuracy alone is misleading when churn is rare. The model is not ranking or identifying new-cohort churners effectively.

### Business calculations

Estimated customer lifetime value is calculated as:

```text
clv_est = Balance + 0.5 * Outstanding Loans + 200 * NumOfProducts
```

Expected lost value is calculated as:

```text
expected_lost_value = churn_score * clv_est
```

The retention policy targets the highest-risk 20% of customers by churn score. The action is then selected based on the customer segment.

## 7. Dashboard and Metrics

The dashboard is available in `dashboard/` and is built with React and Vite.

### Dashboard views

- Training cohort.
- Validation cohort.
- Test cohort.
- New customer or database-scored cohort.

### Dashboard filters

Users can filter by:

- Customer segment.
- Recommended retention action.
- Number of products.
- Risk band.
- Customer tenure.
- Balance range.
- Search text.

### Dashboard outputs

The dashboard shows:

- Active cohort size.
- Churn score distribution.
- Customer segment distribution.
- Recommended action distribution.
- Expected lost value.
- Ranked customer prioritization queue.
- Model benchmark metrics.
- Calibration information where labels are available.
- Cumulative gains and expected-loss coverage.
- Feature importance and individual customer drivers.
- Exportable filtered customer views.

### Metric definitions

#### Accuracy

The share of records classified correctly at a 0.50 churn-probability threshold.

```text
accuracy = (true positives + true negatives) / total records
```

Accuracy can be misleading when churn is rare.

#### Precision

Among customers predicted to churn, the share who actually churned.

```text
precision = true positives / (true positives + false positives)
```

#### Recall

Among customers who actually churned, the share identified by the model.

```text
recall = true positives / (true positives + false negatives)
```

#### F1 score

The harmonic mean of precision and recall.

```text
F1 = 2 * precision * recall / (precision + recall)
```

#### ROC-AUC

Measures how well the model ranks churners above non-churners across classification thresholds. A value near 1.0 is strong, while 0.5 is approximately random ranking.

#### PR-AUC

Measures the precision-recall tradeoff. It is especially useful when churn is an imbalanced class.

#### RMSE of probability

The square root of the mean squared difference between the actual churn label and predicted churn probability.

```text
RMSE = sqrt(mean((actual_churn - predicted_probability)^2))
```

For binary probability predictions, this is the square root of the Brier score.

#### Recall@Top-K

Measures the share of all churners captured by contacting only the highest-risk fraction of customers.

```text
Recall@20% = churners in highest-risk 20% / all churners
```

This is one of the most relevant metrics for a retention campaign with limited outreach capacity.

#### Calibration

Compares predicted risk ranges with observed churn rates. A well-calibrated model with a 0.70 predicted probability should represent a group with approximately 70% observed churn, subject to sample size and data quality.

#### SHAP-style feature contributions

XGBoost contribution values are used to show which features pushed an individual customer’s score higher or lower. The dashboard displays the largest individual drivers and aggregate feature importance.

## 8. Improvements That Can Be Made

### Model improvements

- Investigate the very high original AUC and possible synthetic-data artifacts or leakage.
- Retrain with real, time-ordered customer outcomes.
- Use temporal validation rather than only random splits.
- Add probability calibration.
- Tune the decision threshold according to retention cost and contact capacity.
- Optimize ranking directly for expected lost value.
- Evaluate performance by customer segment, geography, product, and communication channel.
- Add model monitoring for drift, missingness, calibration, and top-k recall.

### Data improvements

- Add explicit missingness indicators.
- Define imputation rules using training-only reference statistics.
- Validate ranges for income, balance, credit scores, tenure, and product counts.
- Detect duplicate customers and duplicate snapshots.
- Add data-quality reports for every scoring run.
- Capture actual retention outcomes and eventual churn labels.

### Engineering improvements

- Add automated unit and integration tests for feature parity between training and inference.
- Add database migrations instead of applying one standalone SQL file.
- Add retry logic and failure alerts for scoring jobs.
- Add a scoring-run audit page to the dashboard.
- Add row-level security and least-privilege database roles before handling real customer data.
- Use a private backend API if the dashboard needs live database queries instead of static files.

## 9. Future Incorporations

Future versions could include:

- Real-time or near-real-time database webhooks.
- Customer-level intervention tracking and retention outcomes.
- Uplift modeling to identify customers who are likely to respond to an intervention.
- Treatment-effect estimation for different retention actions.
- Cost-sensitive optimization that includes contact cost and expected retention value.
- Model version comparison and champion/challenger deployment.
- Automated retraining when enough new labeled outcomes are available.
- Drift alerts when feature or score distributions change.
- Role-based dashboard access.
- Audit logging and approval workflows for retention campaigns.
- Data encryption, private networking, and formal banking data governance.
- A live backend API instead of static dashboard files.
- Additional models such as calibrated Logistic Regression, LightGBM, CatBoost, or neural network benchmarks.
- A/B testing of retention actions and measurement of incremental retained value.

## Running the Project

### Run the notebooks

Run the notebooks in order:

1. `data/01_bank_churn_eda_features.ipynb`
2. `data/02_bank_churn_modeling_recommender.ipynb`
3. `data/03_inference_on_new_customers.ipynb`

### Run the dashboard locally

```powershell
cd dashboard
npm install
npm run dev
```

Open the local URL shown by Vite, usually `http://localhost:5173`.

### Run the database workflow locally

After setting `SUPABASE_DB_URL` in the environment:

```powershell
python jobs/seed_csv.py data/robustness_test/synthetic_banking_stress_test.csv
python jobs/score_customers.py
```

The database schema is in `supabase/schema.sql`. The local database details file is intentionally ignored by Git and should never be committed.
