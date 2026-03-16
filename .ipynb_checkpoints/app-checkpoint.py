import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('model/feature_columns.pkl', 'rb') as f:
        features = pickle.load(f)
    return model, scaler, features

model, scaler, feature_cols = load_model()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Predictor")
st.markdown("Enter customer details below to predict churn probability.")
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Demographics")
    gender          = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen  = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner         = st.selectbox("Partner", ["Yes", "No"])
    dependents      = st.selectbox("Dependents", ["Yes", "No"])
    tenure          = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.subheader("📱 Services")
    phone_service   = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines  = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service= st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup   = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection=st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support    = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv    = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies= st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

with col3:
    st.subheader("💳 Billing")
    contract        = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method  = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 119.0, 65.0)
    total_charges   = st.slider("Total Charges ($)", 0.0, 8700.0, monthly_charges * tenure)

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔮 Predict Churn Risk", use_container_width=True):

    # Build input dict
    input_data = {
        'gender':           1 if gender == "Male" else 0,
        'SeniorCitizen':    1 if senior_citizen == "Yes" else 0,
        'Partner':          1 if partner == "Yes" else 0,
        'Dependents':       1 if dependents == "Yes" else 0,
        'tenure':           tenure,
        'PhoneService':     1 if phone_service == "Yes" else 0,
        'MultipleLines':    1 if multiple_lines == "Yes" else 0,
        'OnlineSecurity':   1 if online_security == "Yes" else 0,
        'OnlineBackup':     1 if online_backup == "Yes" else 0,
        'DeviceProtection': 1 if device_protection == "Yes" else 0,
        'TechSupport':      1 if tech_support == "Yes" else 0,
        'StreamingTV':      1 if streaming_tv == "Yes" else 0,
        'StreamingMovies':  1 if streaming_movies == "Yes" else 0,
        'PaperlessBilling': 1 if paperless == "Yes" else 0,
        'MonthlyCharges':   monthly_charges,
        'TotalCharges':     total_charges,
        'InternetService_Fiber optic': 1 if internet_service == "Fiber optic" else 0,
        'InternetService_No':          1 if internet_service == "No" else 0,
        'Contract_One year':           1 if contract == "One year" else 0,
        'Contract_Two year':           1 if contract == "Two year" else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == "Credit card (automatic)" else 0,
        'PaymentMethod_Electronic check':        1 if payment_method == "Electronic check" else 0,
        'PaymentMethod_Mailed check':            1 if payment_method == "Mailed check" else 0,
    }

    input_df = pd.DataFrame([input_data])[feature_cols]

    # Scale numeric columns
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Predict
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]

    # ── Result Display ────────────────────────────────────────────────────────
    st.subheader("Prediction Result")
    r1, r2, r3 = st.columns(3)

    r1.metric("Churn Probability", f"{prob:.1%}")
    risk_label = "🚨 High Risk" if prob >= 0.7 else "⚠️ Medium Risk" if prob >= 0.4 else "✅ Low Risk"
    r2.metric("Prediction", risk_label)
    r3.metric("Confidence", f"{max(prob, 1-prob):.1%}")

    # Risk level
    if prob >= 0.7:
        st.error(f"🚨 HIGH RISK — This customer has a {prob:.1%} probability of churning. Immediate retention action recommended.")
    elif prob >= 0.4:
        st.warning(f"⚠️ MEDIUM RISK — This customer has a {prob:.1%} probability of churning. Monitor closely.")
    else:
        st.success(f"✅ LOW RISK — This customer has only a {prob:.1%} probability of churning.")

    # Recommendations
    st.divider()
    st.subheader("💡 Recommended Actions")

    recommendations = []
    if contract == "Month-to-month":
        recommendations.append("**Offer a contract upgrade** — Month-to-month customers churn at 43%. Offer 1–2 months free to switch to an annual plan.")
    if internet_service == "Fiber optic":
        recommendations.append("**Review fiber optic value proposition** — Fiber customers churn at 42%. Consider a loyalty discount or service upgrade.")
    if tenure < 12:
        recommendations.append("**Enroll in onboarding program** — New customers churn most in the first year. Schedule a 30/60/90 day check-in.")
    if online_security == "No" and internet_service != "No":
        recommendations.append("**Offer Online Security add-on** — Customers without security features show higher churn rates.")
    if not recommendations:
        recommendations.append("**Maintain engagement** — This customer is low risk. Focus on upsell opportunities rather than retention.")

    for rec in recommendations:
        st.markdown(f"- {rec}")