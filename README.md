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
