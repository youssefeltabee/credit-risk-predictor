import numpy as np
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

np.random.seed(42)
n = 50000

credit_score = np.random.normal(650, 80, n).clip(300, 850)
annual_income = np.random.lognormal(mean=11.0, sigma=0.6, size=n)
dti_ratio = np.random.beta(2, 5, n) * 60
loan_amount = np.random.lognormal(mean=9.5, sigma=0.8, size=n)
loan_term = np.random.choice([12, 24, 36, 48, 60], size=n, p=[0.1, 0.15, 0.35, 0.25, 0.15])
employment_length = np.random.choice([0, 1, 2, 3, 5, 7, 10, 15, 20], size=n, p=[0.1, 0.1, 0.12, 0.15, 0.13, 0.12, 0.1, 0.1, 0.08])
num_credit_lines = np.random.poisson(5, n).clip(1, 30)
delinquent_history = np.random.binomial(1, 0.15, n)
num_delinquencies = delinquent_history * np.random.poisson(2, n).clip(1, 10)
num_credit_inquiries = np.random.poisson(1.5, n).clip(0, 10)
revolving_util = np.random.beta(2, 3, n) * 100

interest_rate = (
    3.0
    - 0.003 * (credit_score - 300)
    + 1.5 * dti_ratio / 60
    + 1.2 * (loan_amount / annual_income)
    + 0.5 * (loan_term / 12)
    - 0.1 * employment_length
    + 2.0 * delinquent_history
    + 0.3 * num_credit_inquiries
    + 0.02 * revolving_util
    + np.random.normal(0, 0.5, n)
).clip(2.0, 30.0)

credit_score_s = (credit_score - 650) / 80
dti_ratio_s = (dti_ratio - 20) / 12
log_lti = np.log1p(loan_amount / annual_income)
lti_s = (log_lti - log_lti.mean()) / log_lti.std()
emp_s = (employment_length - 5) / 6
num_cl_s = (num_credit_lines - 5) / 3
inq_s = (num_credit_inquiries - 1.5) / 1.5
rev_util_s = (revolving_util - 40) / 25

logit = (
    -2.5 * credit_score_s
    + 1.2 * dti_ratio_s
    + 0.8 * lti_s
    - 0.5 * emp_s
    + 0.3 * num_cl_s
    + 2.0 * delinquent_history
    + 0.5 * inq_s
    + 0.4 * rev_util_s
    + 0.3 * ((loan_term - 36) / 18)
    - 2.5
)
prob_default = 1 / (1 + np.exp(-logit))
default = np.random.binomial(1, prob_default)

loan_status = np.where(default == 1, 'Default', 'Fully Paid')

df = pd.DataFrame({
    'credit_score': credit_score.astype(int),
    'annual_income': annual_income.round(2),
    'dti_ratio': dti_ratio.round(2),
    'loan_amount': loan_amount.round(2),
    'loan_term': loan_term,
    'employment_length': employment_length,
    'num_credit_lines': num_credit_lines,
    'delinquent_history': delinquent_history,
    'num_delinquencies': num_delinquencies,
    'num_credit_inquiries': num_credit_inquiries.astype(int),
    'revolving_util': revolving_util.round(2),
    'interest_rate': interest_rate.round(2),
    'loan_status': loan_status,
})

path = 'data/credit_risk_data.csv'
df.to_csv(path, index=False)
print(f'Generated {len(df)} rows, {df.shape[1]} columns')
print(f'Default rate: {df.loan_status.eq("Default").mean():.2%}')
print(f'Saved to {path}')
