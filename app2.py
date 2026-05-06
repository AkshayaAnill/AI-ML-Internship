import streamlit as st
import joblib
import numpy as np


model = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

st.title("Customer Segmentation System")

st.write("Enter customer details:")

monthly_fee = st.number_input("Monthly Fee")
usage = st.number_input("Avg Weekly Usage Hours")
tickets = st.number_input("Support Tickets")
failures = st.number_input("Payment Failures")
tenure = st.number_input("Tenure (months)")
last_login = st.number_input("Days Since Last Login")

if st.button("Predict Cluster"):
    data = scaler.transform([[monthly_fee, usage, tickets, failures, tenure, last_login]])
    cluster = model.predict(data)

    st.success(f"Customer belongs to Cluster: {cluster[0]}")