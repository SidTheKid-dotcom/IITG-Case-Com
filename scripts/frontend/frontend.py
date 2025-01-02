import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the saved ML model

# Load the trained model
model = joblib.load("../../output/models/churn_model_enhanced.pkl")

# Load dataset
dataset_path = '../../input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
data = pd.read_csv(dataset_path)

st.title("Customer Churn Prediction")

# Create sliders for input features
age = st.slider("Customer Age", int(data['CustomerAge'].min()), int(data['CustomerAge'].max()), step=1)
tenure = st.slider("Tenure (months)", int(data['Tenure'].min()), int(data['Tenure'].max()), step=1)
service_usage = st.slider("Service Usage Rate", int(data['ServiceUsageRate'].min()), int(data['ServiceUsageRate'].max()), step=1)
support_calls = st.slider("Support Calls", int(data['SupportCalls'].min()), int(data['SupportCalls'].max()), step=1)
billing_delay = st.slider("Billing Delay Incidents", int(data['BillingDelay'].min()), int(data['BillingDelay'].max()), step=1)
total_expenditure = st.slider("Total Expenditure ($)", int(data['TotalExpenditure'].min()), int(data['TotalExpenditure'].max()), step=10)

# Prepare input for prediction
input_features = np.array([[age, tenure, service_usage, support_calls, billing_delay, total_expenditure]])

# Predict churn probability
churn_prob = model.predict_proba(input_features)[0][1]

# Display the result
st.write(f"Probability of Churn: {churn_prob * 100:.2f}%")
