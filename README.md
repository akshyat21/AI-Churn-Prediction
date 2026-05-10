# 📉 Customer Churn Prediction with AI Retention Advice

**Live Demo:** [https://ai-churn-prediction-hpudyompmu5v7jnxk4epso.streamlit.app/](https://ai-churn-prediction-hpudyompmu5v7jnxk4epso.streamlit.app/)

A production‑ready machine learning system that predicts telecom customer churn and generates personalised, AI‑driven retention recommendations.

---

## 🎯 Business Problem

A telecom company wants to reduce customer churn. The retention team needs to:
- Identify customers **likely to cancel** their subscription  
- Understand **why** those customers are at risk  
- Receive **actionable, tailored recommendations** to retain them  

This project solves that by delivering an end‑to‑end churn prediction system with an interactive Streamlit dashboard and an LLM‑powered retention advisor.

---

## 🧠 Model & Key Decisions

### 1. Algorithm Choice: Logistic Regression
- **Why not XGBoost / Random Forest?**  
  Logistic Regression provides **interpretable coefficients**, which are essential for explaining *why* a customer is at risk. It also trains quickly and is less prone to overfitting with the selected features.
- Performance (AUC = 0.84, recall = 0.77, precision = 0.52) is sufficient for the business use case. The threshold can be adjusted to favour recall or precision as needed.

### 2. Feature Selection (11 features out of 31 dummies)
- Started with all one‑hot encoded columns, then used **correlation with target** and **coefficient magnitude** to select the most predictive features.
- Removed noisy or redundant features (e.g., `gender`, `PhoneService`, low‑impact streaming services) to improve generalisation and reduce overfitting.
- The final feature set balances predictive power and interpretability.

### 3. Handling Class Imbalance (26% churn)
- Instead of oversampling (SMOTE) or class weighting, we used **threshold tuning** to optimise the business trade‑off.  
- Decision threshold set to **0.29** gives recall 0.77 (catch 77% of actual churners) with precision 0.52 (about half of alerts are correct).  
- Alternative thresholds are provided in the code for different business priorities.

### 4. LLM Integration (Groq Llama 3.3)
- The model’s **coefficients** are used to compute feature contributions for each customer.  
- Top **three positive contributors** (risk factors) are passed to the LLM, along with a customer summary.  
- The LLM is **not asked to diagnose risk** – it only generates retention actions based on the provided risk factors. This makes the output grounded and trustworthy.

### 5. Deployment
- **Streamlit** frontend with dropdowns for user‑friendly input.  
- **Joblib** for model serialisation.  
- Hosted on **Streamlit Cloud** – free, fast, and always available.

---

## 📊 Performance Summary (Test Set)

| Metric | Value |
|--------|-------|
| AUC | 0.84 |
| Recall (churn) | 0.77 |
| Precision (churn) | 0.52 |
| F1 (churn) | 0.62 |

The confusion matrix at threshold 0.29:
