import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

model_path = os.path.join(os.path.dirname(__file__), '..', 'models')

model = joblib.load(os.path.join(model_path, 'xgb_model.pkl'))
scaler = joblib.load(os.path.join(model_path, 'scaler.pkl'))
explainer = joblib.load(os.path.join(model_path, 'shap_explainer.pkl'))

feature_names = ['credit_score', 'annual_income', 'dti_ratio', 'loan_amount',
       'loan_term', 'employment_length', 'num_credit_lines',
       'delinquent_history', 'num_credit_inquiries',
       'revolving_util', 'interest_rate']

st.set_page_config(page_title='Credit Risk Predictor', page_icon='\U0001f3e6', layout='wide')

st.markdown("""
<style>
.app-header { background: linear-gradient(135deg, #0f172a, #1e293b); padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 2rem; }
.app-header h1 { color: white; margin: 0; font-size: 1.8rem; font-weight: 600; }
.app-header p { color: #94a3b8; margin: 0.25rem 0 0 0; font-size: 0.9rem; }
.card { background: #1e293b; border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; }
.card h3 { color: #e2e8f0; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; margin: 0 0 1rem 0; }
.metric-big { font-size: 2.5rem; font-weight: 700; text-align: center; padding: 0.5rem 0; }
.metric-label { text-align: center; color: #94a3b8; font-size: 0.85rem; }
.badge-approve { background: linear-gradient(135deg, #059669, #10b981); color: white; padding: 0.5rem 1.5rem; border-radius: 8px; font-weight: 700; font-size: 1.2rem; text-align: center; }
.badge-deny { background: linear-gradient(135deg, #dc2626, #ef4444); color: white; padding: 0.5rem 1.5rem; border-radius: 8px; font-weight: 700; font-size: 1.2rem; text-align: center; }
div[data-testid="stSlider"] label { font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="app-header"><h1>\U0001f3e6 Credit Risk Predictor</h1><p>Assess loan applicants with ML-powered risk analysis + SHAP explainability</p></div>', unsafe_allow_html=True)

input_col, result_col = st.columns([1, 1.2])

with input_col:
    st.markdown('<div class="card"><h3>Applicant Profile</h3>', unsafe_allow_html=True)
    credit_score = st.slider('Credit Score', 300, 850, 650)
    annual_income = st.number_input('Annual Income ($)', min_value=10000, max_value=500000, value=60000, step=5000)
    dti_ratio = st.slider('DTI Ratio (%)', 0.0, 60.0, 20.0)
    loan_amount = st.number_input('Loan Amount ($)', min_value=1000, max_value=100000, value=15000, step=1000)
    loan_term = st.selectbox('Loan Term (months)', [12, 24, 36, 48, 60])
    employment_length = st.slider('Employment Length (years)', 0, 20, 5)

    st.markdown('<h3>Credit History</h3>', unsafe_allow_html=True)
    num_credit_lines = st.slider('Number of Credit Lines', 1, 30, 5)
    delinquent_history = st.selectbox('Delinquent History', ['No', 'Yes'])
    num_credit_inquiries = st.slider('Credit Inquiries (last 2 years)', 0, 10, 2)
    revolving_util = st.slider('Revolving Utilization (%)', 0.0, 100.0, 30.0)
    interest_rate = st.slider('Interest Rate (%)', 2.0, 30.0, 10.0)
    st.markdown('<h3>Decision Threshold</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color:#94a3b8;font-size:0.8rem;margin:0 0 0.5rem 0">Denying a good applicant = lost interest. Approving a defaulter = lost principal (~50&times; worse).</p>', unsafe_allow_html=True)
    threshold = st.slider('Threshold', 0.05, 0.95, 0.30, 0.05,
        help='Applicants above this risk level are denied. Lower = stricter, fewer defaults but more missed good loans.')
    st.markdown('</div>', unsafe_allow_html=True)

delinquent_history_binary = 1 if delinquent_history == 'Yes' else 0

input_data = np.array([[credit_score, annual_income, dti_ratio, loan_amount,
                        loan_term, employment_length, num_credit_lines,
                        delinquent_history_binary, num_credit_inquiries,
                        revolving_util, interest_rate]])

input_scaled = scaler.transform(input_data)
proba = float(model.predict_proba(input_scaled)[0, 1])

decision = 'DENY' if proba >= threshold else 'APPROVE'

with result_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Risk Assessment</h3>', unsafe_allow_html=True)

    metric_row = st.columns([1, 1, 1])
    with metric_row[0]:
        pct = int(proba * 100)
        st.markdown(f'<div class="metric-big">{pct}%</div><div class="metric-label">Default Probability</div>', unsafe_allow_html=True)
        st.progress(proba)
    with metric_row[1]:
        badge = 'badge-approve' if decision == 'APPROVE' else 'badge-deny'
        st.markdown(f'<div style="margin-top:0.5rem"><div class="{badge}">{decision}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-label">Decision at {threshold:.0%} threshold</div>', unsafe_allow_html=True)
    with metric_row[2]:
        safe_margin = proba - threshold if proba >= threshold else threshold - proba
        st.markdown(f'<div class="metric-big" style="font-size:1.8rem">{safe_margin:.1%}</div><div class="metric-label">Margin from threshold</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>SHAP Explanation — What drove this decision?</h3>', unsafe_allow_html=True)

    shap_values = explainer(input_scaled)
    shap_values.feature_names = feature_names

    fig, ax = plt.subplots(figsize=(10, 4.5))
    shap.plots.waterfall(shap_values[0], max_display=12, show=False)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    for text in ax.get_yticklabels() + ax.get_xticklabels():
        text.set_color('#e2e8f0')
    ax.set_xlabel('SHAP value (impact on model output)', color='#94a3b8')
    ax.spines['bottom'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.tick_params(colors='#94a3b8')
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.caption('Red pushes toward Default (deny). Blue pushes toward Fully Paid (approve).')
    st.markdown('</div>', unsafe_allow_html=True)
