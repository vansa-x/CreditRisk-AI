import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
import io
from src.model import calculate_roi_metrics, generate_shap_explainer

# --- Load Real Model & Scaler ---
@st.cache_resource
def load_artifacts():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

def generate_narrative_insights(shap_values_instance, input_df):
    """Translates the top 2 SHAP values into a plain-English narrative."""
    vals = shap_values_instance.values
    
    # Get indices of the top 2 features by absolute SHAP value
    top_indices = np.argsort(np.abs(vals))[-2:][::-1]
    
    narratives = []
    for idx in top_indices:
        feat_name = input_df.columns[idx]
        feat_val = input_df.iloc[0, idx]
        shap_val = vals[idx]
        
        # Format the value for readability
        if feat_name in ['GrAppv', 'SBA_Appv']:
            val_str = f"${feat_val:,.0f}"
        elif feat_name == 'NewExist':
            val_str = "New Startup" if feat_val == 2.0 else "Existing Business"
        else:
            val_str = f"{feat_val:,.0f}"
            
        if shap_val > 0:
            narratives.append(f"🔴 **Increased Risk:** The {feat_name} of **{val_str}** drove the risk score higher.")
        else:
            narratives.append(f"🟢 **Reduced Risk:** The {feat_name} of **{val_str}** significantly lowered the default risk.")
            
    return "\n".join(narratives)

st.set_page_config(page_title="CreditRisk AI Dashboard", layout="wide")

# --- UI Enhancement: Custom CSS ---
st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #2b2b2b;
        border: 1px solid #444444;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #B0B0B0;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.title("CreditRisk AI: Commercial Underwriting Engine")

# --- Sidebar Inputs ---
st.sidebar.header("Applicant Financials")
term = st.sidebar.slider("Loan Term (Months)", min_value=12, max_value=360, value=120)
gr_appv = st.sidebar.number_input("Gross Approved Amount ($)", min_value=10000, max_value=5000000, value=500000, step=10000)
guarantee_pct = st.sidebar.slider("SBA Guarantee Percentage", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
sba_appv = gr_appv * guarantee_pct
no_emp = st.sidebar.number_input("Number of Employees", min_value=1, max_value=500, value=10)
retained_job = st.sidebar.number_input("Jobs Retained", min_value=0, max_value=500, value=10)
new_exist = st.sidebar.radio("Business Status", options=[1.0, 2.0], format_func=lambda x: "Existing Business" if x == 1.0 else "New Startup")
interest_rate = st.sidebar.slider("Estimated Interest Rate", min_value=0.01, max_value=0.20, value=0.08, step=0.01)

input_features = pd.DataFrame({
    'Term': [term], 'GrAppv': [gr_appv], 'SBA_Appv': [sba_appv],
    'NoEmp': [no_emp], 'RetainedJob': [retained_job], 'NewExist': [new_exist]
})

try:
    model, scaler = load_artifacts()
    
    scaled_features_array = scaler.transform(input_features)
    probability_of_default = float(model.predict_proba(scaled_features_array)[0][1])
    roi_metrics = calculate_roi_metrics(probability_of_default, gr_appv, guarantee_pct, interest_rate)
    
    # --- Top-Level KPI Scorecards ---
    st.subheader("Executive Decision & ROI Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Recommendation", roi_metrics["Recommendation"], 
                delta="High Risk" if roi_metrics["Recommendation"] == "Reject" else "Cleared", delta_color="inverse")
    col2.metric("Net Expected Value", f"${roi_metrics['Net Expected Value']:,.2f}")
    col3.metric("Expected Revenue", f"${roi_metrics['Expected Revenue']:,.2f}")
    col4.metric("Expected Loss", f"${roi_metrics['Expected Loss']:,.2f}")
    
    st.markdown("---")
    st.write(f"**Calculated Probability of Default (PD):** {probability_of_default:.1%}")
    st.progress(probability_of_default)
    
    # --- Regulatory Audit Trail & Data Storytelling ---
    st.markdown("---")
    st.subheader("Underwriting Audit: Executive Insights")
    
    explainer = generate_shap_explainer(model)
    shap_values = explainer(scaled_features_array) 
    shap_values.feature_names = list(input_features.columns) 
    
    # Generate and display the narrative
    narrative_text = generate_narrative_insights(shap_values[0], input_features)
    st.info(narrative_text)
    
    # Render the transparent SHAP chart
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    shap.plots.waterfall(shap_values[0], show=False)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", dpi=300)
    st.image(buf)

except Exception as e:
    st.error(f"Error loading model or making prediction: {e}")