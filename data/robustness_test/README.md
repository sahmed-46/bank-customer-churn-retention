# Model robustness test

This folder contains a reproducible, labeled synthetic banking stress test for the saved XGBoost model.

Run from the repository root with the project Python environment:

```powershell
C:\Users\syedo\anaconda3\python.exe data\robustness_test\generate_and_evaluate.py
```

The generator creates 10,000 records with skewed financial variables, heavy-tail outliers, missing values, class imbalance, categorical variation, population drift, and concept drift. The evaluation uses the same feature engineering and model feature alignment as the project pipeline.

The generated cohort is deliberately a robustness test, not a substitute for validation on real future customer outcomes. RMSE is computed on churn probabilities and is equivalent to the square root of the Brier score. Classification metrics use a 0.50 probability threshold.

The workbook can be rebuilt with the spreadsheet builder after the CSV results exist:

```powershell
node data\robustness_test\build_results_workbook.mjs
```
