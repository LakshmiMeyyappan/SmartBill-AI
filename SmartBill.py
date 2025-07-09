
# ----- Imports & paths --------------------------------------------

from pathlib import Path
import streamlit as st
import pandas as pd, joblib, shap
import matplotlib.pyplot as plt
import seaborn as sns

# ---------- Load Trained Model & Data -----------------------------

MODEL_PATH = Path("model.pkl")
model      = joblib.load(MODEL_PATH)          # scikit‑learn pipeline
explainer  = shap.Explainer(model)

data       = pd.read_csv("electricity_bill_dataset.csv") \
               .drop(columns=["Company", "City"])
feature_medians = data.median(numeric_only=True)

# ------------Select Top‑K Most Important Features -----------------

k = 5  # ← change this if you want more/less inputs
importances      = pd.Series(model.feature_importances_,index=model.feature_names_in_)
top_features     = importances.sort_values(ascending=False).head(k).index

#------------------Widget Map----------------------------------

# Dictionary of lambdas → each key returns a ready‑made Streamlit widget.

_WIDGET_MAP = {
    "Fan":            lambda: st.number_input("Fans",               0,10,1),
    "Refrigerator":   lambda: st.number_input("Refrigerators",      0,10,1),
    "AirConditioner": lambda: st.number_input("AC units",           0,10,1),
    "Television":     lambda: st.number_input("Televisions",        0,10,1),
    "Month":          lambda : st.slider("Month (1‑12)",            1,12,1),
    "MonthlyHours":   lambda: st.number_input("Monthly usage hours",1,1000,400),
    "TariffRate":     lambda: st.number_input("Tariff rate (₹/kWh)",1.0, 20.0, 8.0),
}

def widget_for(feature):
    """
    Render the zero‑default widget for `feature`.
    Falls back to 0 if the feature is not in the map.
    """
    return _WIDGET_MAP.get(feature, lambda: 0)()


# -------------- Streamlit page ----------------------------------

st.set_page_config(page_title="SmartBill AI")
st.title(f"**SmartBill AI – Electricity Bill Predictor**")
st.header("Enter your appliance usage")
st.markdown(f"This AI-powered tool predicts your electricity bill using the {k} most influential features.")

#-----------------Gather Inputs-------------------------------

# Collect user inputs for top‑k features, others use median defaults

row = pd.Series(0, index=model.feature_names_in_)  # start with zeros for every feature

for feat in top_features:                         
    row[feat] = widget_for(feat)

#----------------Predict & Local SHAP Explanation---------------

# SHAP generates a waterfall chart so users see how each chosen input pushed the estimate up or down.
if st.button("Predict"):

    bill = float(model.predict(pd.DataFrame([row]))[0])
    st.metric("Estimated Bill", f"₹{bill:,.0f}")

    shap_values = explainer(pd.DataFrame([row]))
    st.subheader("SHAP Waterfall – Feature Contribution")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())

#--------------Sidebar – Global Feature Importance------------

#Horizontal bar plot of every feature’s relative importance.
with st.sidebar:
    if st.checkbox("Show global feature importance"):
        st.subheader("Global Feature Importance")
        fig, ax = plt.subplots(figsize=(6, 0.4 * len(importances)))
        sns.barplot(x=importances.values, y=importances.index, palette="viridis", ax=ax, orient="h")
        st.pyplot(fig)
        
#------------Sidebar – Performance Scatter Plot-------------
    
#Shows actual vs predicted for the entire training set.
    if st.sidebar.checkbox("Show model performance plot"):
        st.sidebar.info("Generating scatter plot …")
    
        independent = data.drop("ElectricityBill", axis=1)
        dependent = data["ElectricityBill"]
        y_pred_full = model.predict(independent)
    
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(dependent, y_pred_full, alpha=0.35, label="Training data")
        lims = [min(dependent.min(), y_pred_full.min()), max(dependent.max(), y_pred_full.max())]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    
        # Safely plot user's prediction only if it exists
        if 'bill' in locals():
            ax.scatter(
                x=[bill], y=[bill],
                color="red", s=80, marker="*", label="This household"
            )
    
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual Bill (₹)")
        ax.set_ylabel("Predicted Bill (₹)")
        ax.set_title("Gradient Boosting Regressor – Actual vs Predicted")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(frameon=False)
    
        st.pyplot(fig)

st.markdown("---")
st.caption("Made by Lakshmi Meyyappan • HOPE AI Hackathon 2025")