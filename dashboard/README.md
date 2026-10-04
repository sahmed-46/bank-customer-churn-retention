# Bank Retention Signals

React dashboard for exploring the project's saved churn scores, customer value estimates, retention actions, model evaluation results, and model explanations.

## Run locally

From this directory:

```bash
npm install
npm run dev
```

`npm run build` creates a static production build in `dist/`. Asset paths are relative, so the build works from any subfolder (including GitHub Pages).

## Data

The dashboard reads pre-built files from `public/data/` rather than the raw CSVs:

| File | Contents |
|---|---|
| `manifest.json` | Column layout, category labels, feature labels, partition boundaries, partition-level SHAP importance |
| `labeled.bin`, `inference.bin` | Row-level scores, value estimates, segments, actions, and labels (loaded on start) |
| `labeled-drivers.bin`, `inference-drivers.bin` | Top three SHAP contributions per customer (loaded when a customer is selected) |

Regenerate them after re-running the notebooks:

```bash
python scripts/build_data.py
```

The script rebuilds the model features from `../data/*.csv` exactly as notebooks 01–03 do, and stops if the saved XGBoost model no longer reproduces the scores in `../data/outputs_bank/`. It needs `pandas`, `numpy`, `scikit-learn`, and `xgboost`.

## Data views

- Training (60,000 rows), validation (20,000), and test (20,000) are slices of `bank_churn_recommendations.csv`. The notebook concatenates these in that order; the boundaries come from re-running the notebook's stratified split.
- The training view is in-sample and should not be interpreted as held-out performance.
- New customers (15,640 rows) comes from `new_customers_churn_predictions.csv`; it has predictions and recommended actions but no labeled outcome, so calibration is not shown for it.
- Customer names, contact details, addresses, and IDs are not loaded into the interface.

Validation/test model comparison values are the saved results from `02_bank_churn_modeling_recommender.ipynb`. Cohort filters recalculate metrics, calibration, and gains from the selected partition.

## Model explainability

- **What drives churn risk**: mean absolute SHAP value per feature over the whole partition (not filtered).
- **Calibration**: observed churn rate per 10-point score bin against the diagonal, plus the Brier score.
- **Cumulative gains**: share of churners and of expected lost value reached when contacting customers in score order.
- **Customer drivers**: selecting a row in the prioritization queue shows the three features that moved that customer's score the most, in log-odds.

## Deploy

`.github/workflows/deploy-dashboard.yml` builds and publishes `dist/` to GitHub Pages on pushes to `main` that touch `dashboard/`. Enable it once under **Settings → Pages → Source: GitHub Actions**.
