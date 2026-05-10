import os
import joblib
import json
import numpy as np
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (GROQ_API_KEY)
load_dotenv()

# Load model and artifacts
model = joblib.load('churn_model.pkl')
with open('selected_features.json', 'r') as f:
    selected_features = json.load(f)
with open('churn_threshold.txt', 'r') as f:
    threshold = float(f.read())

# Get coefficients for interpretation (used to identify risky features)
coefficients = model.coef_[0]
feature_coeffs = dict(zip(selected_features, coefficients))

# Prompt template for Groq LLM
prompt = PromptTemplate(
    input_variables=["probability", "threshold", "risky_features", "customer_summary"],
    template="""You are a customer retention AI for a telecom company.

A customer has a churn probability of {probability:.2%} (threshold = {threshold:.0%}).

Key risk factors contributing to this probability:
{risky_features}

Customer snapshot: {customer_summary}

Based on the above, provide 2-3 specific, actionable recommendations to retain this customer. Focus on:
- Contract type (e.g., offer a 1‑year discount if on month‑to‑month)
- Missing or risky services (e.g., upgrade to fibre, add online security)
- Price sensitivity (e.g., loyalty discount, bundle offer)

Write in a professional, helpful tone.

Recommendations:
1.
2.
3.
"""
)

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)
chain = prompt | llm | StrOutputParser()

def get_retention_advice(customer_data_dict, probability):
    """
    customer_data_dict: dict with keys matching selected_features (values numeric)
    probability: churn probability from model (float between 0 and 1)
    Returns: string with recommendations
    """
    # Compute per‑feature contributions to churn risk (positive only)
    contributions = {}
    for feat in selected_features:
        val = customer_data_dict.get(feat, 0)
        contrib = feature_coeffs[feat] * val
        if contrib > 0:
            contributions[feat] = contrib
    # Sort and get top 3 risky features
    top_risky = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:3]
    risky_str = "\n".join([f"- {feat}: contribution {contrib:.3f}" for feat, contrib in top_risky])
    if not risky_str:
        risky_str = "- No strong risk factors (probability near threshold)."
    
    # Create a simple customer summary (customize as needed)
    tenure = customer_data_dict.get('tenure', 0)
    monthly = customer_data_dict.get('MonthlyCharges', 0)
    contract = "month‑to‑month"
    if customer_data_dict.get('Contract_One year', 0) == 1:
        contract = "one‑year contract"
    elif customer_data_dict.get('Contract_Two year', 0) == 1:
        contract = "two‑year contract"
    summary = f"{tenure} months tenure, monthly charges {monthly:.0f}, {contract}, "
    # Add flags for common services
    if customer_data_dict.get('InternetService_Fiber optic', 0):
        summary += "fiber internet, "
    if customer_data_dict.get('OnlineSecurity_Yes', 0):
        summary += "has online security, "
    else:
        summary += "lacks online security, "
    if not customer_data_dict.get('TechSupport_Yes', 0):
        summary += "no tech support."
    
    response = chain.invoke({
        "probability": probability,
        "threshold": threshold,
        "risky_features": risky_str,
        "customer_summary": summary
    })
    return response