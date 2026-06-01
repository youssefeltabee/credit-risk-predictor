# Credit Risk Predictor

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-0.923%20AUC-EC1C24)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit)
![SHAP](https://img.shields.io/badge/SHAP-0.51-2E86AB)
![License](https://img.shields.io/badge/License-MIT-green)

ML-powered loan application risk assessment with SHAP explainability.

## Overview

Predict whether a loan applicant will default using XGBoost, then explain *why* the model made its decision with SHAP. Built on synthetic credit data (50K applicants, 22.78% default rate) with realistic feature relationships.

**Performance**: ROC-AUC ~0.92, Default F1 ~0.72.

## Project Structure

```
src/generate_data.py              # Synthetic data generator
src/train_model.py                 # Standalone training script
dashboard/dashboard.py             # Streamlit dashboard
notebooks/01_eda_preprocessing.ipynb # EDA, scaling, SMOTE
notebooks/02_modeling_shap.ipynb    # XGBoost training + SHAP analysis
models/xgb_model.pkl               # Trained XGBoost model
models/scaler.pkl                  # Fitted StandardScaler
models/shap_explainer.pkl          # SHAP TreeExplainer
.gitignore
requirements.txt
README.md
```

## Quick Start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_eda_preprocessing.ipynb
jupyter notebook notebooks/02_modeling_shap.ipynb
python src/train_model.py
streamlit run dashboard/dashboard.py
```

## Key Results

| Metric | Value |
|--------|-------|
| ROC-AUC | 0.923 |
| Accuracy | 87% |
| Default F1 | 0.72 |

**Top predictors**: credit_score, dti_ratio, delinquent_history, interest_rate.

Higher credit scores drive SHAP values downward (toward approval). Delinquent history drives them upward (toward default).

## Decision Threshold

The dashboard defaults to a 30% threshold (not 50%) because approving a defaulter costs ~50x more than denying a good applicant. Adjustable via slider.
