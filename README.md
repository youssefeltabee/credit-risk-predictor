# Credit Risk Predictor

ML-powered loan application risk assessment with SHAP explainability.

## Overview

Predict whether a loan applicant will default using XGBoost, then explain *why* the model made its decision with SHAP. Built on synthetic credit data (50K applicants, 22.78% default rate) with realistic feature relationships.

**Performance**: ROC-AUC ~0.92, Default F1 ~0.72.

## Project Structure

```
credit-risk-predictor/
├── data/
│   └── credit_risk_data.csv      # Generated dataset (50K rows)
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb # EDA, scaling, SMOTE
│   └── 02_modeling_shap.ipynb    # XGBoost training + SHAP analysis
├── src/
│   ├── generate_data.py           # Synthetic data generator
│   └── train_model.py             # Standalone training script
├── dashboard/
│   └── dashboard.py               # Streamlit dashboard
├── models/
│   ├── xgb_model.pkl              # Trained XGBoost model
│   ├── scaler.pkl                 # Fitted StandardScaler
│   └── shap_explainer.pkl         # SHAP TreeExplainer
├── .gitignore
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt

# Run notebooks to recreate from scratch
jupyter notebook notebooks/01_eda_preprocessing.ipynb
jupyter notebook notebooks/02_modeling_shap.ipynb

# Or just train directly
python src/train_model.py

# Launch dashboard
streamlit run dashboard/dashboard.py
```

## Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.923 |
| Accuracy | 87% |
| Default F1 | 0.72 |

**Top predictors**: credit_score, dti_ratio, delinquent_history, interest_rate.

Higher credit scores drive SHAP values downward (toward approval). Delinquent history drives them upward (toward denial).

## Decision Threshold

The dashboard defaults to a 30% threshold (not 50%) because approving a defaulter costs ~50x more than denying a good applicant. Adjustable via slider.
