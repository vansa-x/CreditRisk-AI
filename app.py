import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from src.model import calculate_roi_metrics, generate_shap_explainer

# --- Load Real Model & Scaler ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

st.set_page_config(page_title="CreditRisk AI Dashboard", layout="wide")
st.title("CreditRisk AI: Commercial Underwriting Engine")

# --- Sidebar: Live Inference Inputs ---
st.sidebar.header("Applicant Financials")
st.sidebar.markdown("Adjust parameters for live inference.")

# Inputs mapped exactly to the 6 features our model expects
term = st.sidebar.slider("Loan Term (Months)", min_value=12, max_value=360, value=120)
gr_appv = st.sidebar.number_input("Gross Approved Amount ($)", min_value=10000, max_value=5000000, value=500000, step=10000)
guarantee_pct = st.sidebar.slider("SBA Guarantee Percentage", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
sba_appv = gr_appv * guarantee_pct # Calculate SBA_Appv based on gross and percentage
no_emp = st.sidebar.number_input("Number of Employees", min_value=1, max_value=500, value=10)
retained_job = st.sidebar.number_input("Jobs Retained", min_value=0, max_value=500, value=10)
new_exist = st.sidebar.radio("Business Status", options=[1.0, 2.0], format_func=lambda x: "Existing Business" if x == 1.0 else "New Startup")
interest_rate = st.sidebar.slider("Estimated Interest Rate", min_value=0.01, max_value=0.20, value=0.08, step=0.01)

# Compile features into a DataFrame in the EXACT order the model expects
input_features = pd.DataFrame({
    'Term': [term],
    'GrAppv': [gr_appv],
    'SBA_Appv': [sba_appv],
    'NoEmp': [no_emp],
    'RetainedJob': [retained_job],
    'NewExist': [new_exist]
})

try:
    model, scaler = load_artifacts()
    
    # 1. Scale the live inputs exactly like the training data
    scaled_features_array = scaler.transform(input_features)
    
    # 2. Predict Probability of Default in real-time
    probability_of_default = float(model.predict_proba(scaled_features_array)[0][1])
    
    # 3. Calculate ROI metrics for the executive view
    roi_metrics = calculate_roi_metrics(probability_of_default, gr_appv, guarantee_pct, interest_rate)
    
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
    shap_values = explainer(scaled_features_array) 
    shap_values.feature_names = list(input_features.columns) # Add feature names for the chart
    
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

except Exception as e:
    st.error(f"Error loading model or making prediction: {e}")