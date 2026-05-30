import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/credit_risk_data.csv')

X = df.drop(columns=['loan_status', 'num_delinquencies'])
y = (df['loan_status'] == 'Default').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_res, y_train_res)

joblib.dump(model, 'models/xgb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

explainer = shap.TreeExplainer(model)
joblib.dump(explainer, 'models/shap_explainer.pkl')

from sklearn.metrics import roc_auc_score, classification_report
print('Model AUC:', roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1]))
print(classification_report(y_test, model.predict(X_test_scaled)))
