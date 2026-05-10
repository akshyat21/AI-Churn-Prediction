# app.py
import streamlit as st
import pandas as pd
import joblib
import json
from churn_llm import get_retention_advice

# Load model and threshold
model = joblib.load('churn_model.pkl')
with open('churn_threshold.txt', 'r') as f:
    threshold = float(f.read())

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("📉 Customer Churn Predictor with AI Retention Advice")
st.markdown("Enter customer details to predict churn risk and get actionable retention recommendations.")

with st.form("customer_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12, step=1)
        monthly_charges = st.number_input("Monthly Charges (AED)", min_value=0.0, max_value=200.0, value=70.0, step=5.0)
        
        # Contract type dropdown
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        
        # Internet service
        internet = st.selectbox("Internet Service", ["DSL / Other", "Fiber optic"])
        
        # Payment method
        payment = st.selectbox("Payment Method", ["Electronic check", "Other"])
        
        # Paperless billing
        paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    
    with col2:
        # Online security
        online_security = st.selectbox("Online Security", ["No", "Yes"])
        
        # Tech support
        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        
        # Dependents
        dependents = st.selectbox("Has dependents", ["No", "Yes"])
        
        # Senior citizen
        senior_citizen = st.selectbox("Senior citizen", ["No", "Yes"])
    
    submitted = st.form_submit_button("Predict Churn")

if submitted:
    # Map selections to model features
    # Contract
    contract_one = 1 if contract == "One year" else 0
    contract_two = 1 if contract == "Two year" else 0
    
    # Internet (only fiber optic is positive)
    internet_fiber = 1 if internet == "Fiber optic" else 0
    
    # Payment method
    payment_electronic = 1 if payment == "Electronic check" else 0
    
    # Binary features
    paperless_binary = 1 if paperless == "Yes" else 0
    online_security_binary = 1 if online_security == "Yes" else 0
    tech_support_binary = 1 if tech_support == "Yes" else 0
    dependents_binary = 1 if dependents == "Yes" else 0
    senior_binary = 1 if senior_citizen == "Yes" else 0
    
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract_One year': contract_one,
        'Contract_Two year': contract_two,
        'InternetService_Fiber optic': internet_fiber,
        'PaymentMethod_Electronic check': payment_electronic,
        'PaperlessBilling': paperless_binary,
        'OnlineSecurity_Yes': online_security_binary,
        'TechSupport_Yes': tech_support_binary,
        'Dependents': dependents_binary,
        'SeniorCitizen': senior_binary
    }
    
    # Predict
    input_df = pd.DataFrame([input_dict])
    prob = model.predict_proba(input_df)[0][1]
    
    st.metric("Churn Probability", f"{prob:.1%}")
    
    if prob >= threshold:
        st.warning("⚠️ High churn risk. AI generating retention recommendations...")
        with st.spinner("Consulting AI engine..."):
            advice = get_retention_advice(input_dict, prob)
        st.success("Retention Plan")
        st.markdown(advice)
    else:
        st.success("✅ Low churn risk. The customer is likely to stay.")