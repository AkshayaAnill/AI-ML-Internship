import streamlit as st
import numpy as np
import joblib
model = joblib.load("linear_model.pkl")
st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💼 Salary Prediction App")
st.markdown("Predict salary based on years of experience")
st.sidebar.header("Input Details")
years = st.sidebar.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)
if st.sidebar.button("Predict Salary"):
    prediction = model.predict([[years]])

    st.success(f"Estimated Salary: ₹ {prediction[0]:,.2f}")
