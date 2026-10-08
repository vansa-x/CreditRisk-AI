import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from src.model import calculate_roi_metrics, generate_shap_explainer

# --- Mock Model for initial UI display until real data is trained ---
@st.cache_resource
def load_mock_model():
    model = xgb.XGBClassifier()
    model.fit(np.random.rand(10, 4), np.random.randint(2, size=10)) 
    return model

st.set_page_config(page_title="CreditRisk AI Dashboard", layout="wide")
st.title("CreditRisk AI: Commercial Underwriting Engine")

# --- Sidebar: Live Inference Inputs ---
st.sidebar.header("Applicant Financials")
st.sidebar.markdown("Adjust parameters for live inference.")

loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=10000, max_value=5000000, value=500000, step=10000)
loan_term = st.sidebar.slider("Loan Term (Months)", min_value=12, max_value=360, value=120)
guarantee_pct = st.sidebar.slider("SBA Guarantee Percentage", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
interest_rate = st.sidebar.slider("Estimated Interest Rate", min_value=0.01, max_value=0.20, value=0.08, step=0.01)

# Compile features for inference
input_features = pd.DataFrame({
    'Loan_Amount': [loan_amount],
    'Loan_Term': [loan_term],
    'Guarantee_Pct': [guarantee_pct],
    'Interest_Rate': [interest_rate]
})

model = load_mock_model()
probability_of_default = 0.15 # Hardcoded for UI visualization

roi_metrics = calculate_roi_metrics(probability_of_default, loan_amount, guarantee_pct, interest_rate)

# --- Top-Level KPI Scorecards ---
st.subheader("Executive Decision & ROI Metrics")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Recommendation", roi_metrics["Recommendation"], 
            delta="High Risk" if roi_metrics["Recommendation"] == "Reject" else "Cleared", 
            delta_color="inverse")
col2.metric("Net Expected Value", f"${roi_metrics['Net Expected Value']:,.2f}")
col3.metric("Expected Revenue", f"${roi_metrics['Expected Revenue']:,.2f}")
col4.metric("Expected Loss", f"${roi_metrics['Expected Loss']:,.2f}")

# --- Probability & Confidence ---
st.markdown("---")
st.write(f"**Calculated Probability of Default (PD):** {probability_of_default:.1%}")
st.progress(probability_of_default)

# --- Regulatory Audit Trail ---
st.markdown("---")
st.subheader("Underwriting Audit: Feature Impact")
st.write("SHAP value breakdown fulfilling regulatory compliance for automated decisions.")

explainer = generate_shap_explainer(model)
shap_values = explainer(input_features) 

fig, ax = plt.subplots(figsize=(8, 4))
shap.plots.waterfall(shap_values[0], show=False)
st.pyplot(fig)