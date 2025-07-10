import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Load model and selected features
# -------------------------------
model = pickle.load(open("finalized_model.sav", "rb"))
selected_features = pickle.load(open("selected_features.sav", "rb"))

# -------------------------------
# Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="SmartBill - Electricity Predictor", layout="centered")
st.title("SmartBill: Electricity Bill Predictor")
st.markdown("Predict your monthly electricity bill and understand the prediction with SHAP explainability.")

# -------------------------------
# Input Section
# -------------------------------
user_input = {}
st.subheader(" Enter Appliance Usage and Details")

for feature in selected_features:
    if feature in ['Month']:
        user_input[feature] = st.selectbox(f"{feature}", list(range(1, 13)))
    elif feature == 'City':
        user_input[feature] = st.selectbox(f"{feature}", ['Chennai', 'Hyderabad', 'Bangalore', 'Mumbai'])
    elif feature == 'Company':
        user_input[feature] = st.selectbox(f"{feature}", ['Tata Power Company Ltd.', 'Adani Electricity', 'BESCOM'])
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0)

# -------------------------------
# Prediction + SHAP
# -------------------------------
if st.button(" Predict Bill"):
    input_df = pd.DataFrame([user_input])
    input_df = input_df[selected_features]

    # Prediction
    prediction = model.predict(input_df)[0]
    st.success(f" Predicted Electricity Bill: ₹{prediction:,.2f}")

    # SHAP Explainability
    st.subheader("SHAP Explanation")

    try:
        explainer = shap.Explainer(model)
        shap_values = explainer(input_df)

        # Waterfall Plot
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error("SHAP explanation failed.")
        st.exception(e)
