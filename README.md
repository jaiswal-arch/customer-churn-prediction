# 📊 Customer Churn Prediction

> ML-powered web app that predicts customer churn probability using Logistic Regression — deployed live with actionable business recommendations for each prediction.

🔗 **[Live App](https://customer-churn-prediction-djf7gscnmzfsp3uhwpqxcz.streamlit.app)**

---

## What It Does

A telecom company loses revenue every time a customer cancels their service. This app predicts which customers are at risk of churning — before they leave — so the business can intervene with targeted retention offers.

Enter a customer's demographic, service, and billing details and the app returns:
- **Churn probability** — the likelihood this customer will leave
- **Risk level** — High, Medium, or Low
- **Recommended actions** — specific, data-driven retention strategies

---

## Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.84 |
| Accuracy | 80% |
| Precision (Churn) | 66% |
| Recall (Churn) | 56% |

**Why Logistic Regression over Random Forest:**
Logistic Regression achieved a higher ROC-AUC (0.84 vs 0.82) and is significantly more interpretable — coefficients map directly to business levers like contract type and tenure, making it easier to explain predictions to non-technical stakeholders.

---

## Key Findings from EDA

- **Month-to-month contracts** churn at 43% vs 3% for two-year contracts — the single strongest predictor
- **Fiber optic customers** churn at 42% — nearly double DSL, suggesting a price-value mismatch
- **Senior citizens** churn at 42% vs 24% for non-seniors
- **New customers churn most** — churn spikes sharply in the first 10 months of tenure

---

## Business Impact

With 26.5% churn rate across 7,043 customers and average revenue of $65/month:

- **Revenue at risk:** ~$121,485/month
- **Targeted intervention on top 500 high-risk customers:**
  - Retention offer cost: $10,000
  - Estimated revenue saved: $21,450/month
  - **Net ROI: +$11,450/month**

---

## Tech Stack

| Layer | Tool |
|---|---|
| Modeling | Scikit-learn (Logistic Regression, Random Forest) |
| Analysis | Pandas, NumPy, Matplotlib, Seaborn |
| App | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```
customer-churn-prediction/
├── app.py                        # Streamlit application
├── requirements.txt
├── model/
│   ├── churn_model.pkl           # Trained Logistic Regression model
│   ├── scaler.pkl                # StandardScaler for numeric features
│   └── feature_columns.pkl      # Feature column order
└── notebook/
    └── churn_analysis.ipynb      # Full EDA + model training notebook
```

---

## How to Run Locally

```bash
# Clone the repo
git clone https://github.com/jaiswal-arch/customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Dataset

**Source:** IBM Telco Customer Churn — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

- 7,043 customers
- 21 features — demographics, services, billing
- Binary target: Churn (Yes/No)
- 26.5% churn rate
