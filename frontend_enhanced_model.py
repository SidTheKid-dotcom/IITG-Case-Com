import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the saved ML model

# Load the trained model
model = joblib.load("./output/models/churn_model_enhanced.pkl")

# Load dataset for dynamic slider ranges
dataset_path = './input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
data = pd.read_csv(dataset_path)

st.title("Customer Churn Prediction")

# Create sliders for numerical input features
age = st.slider("Customer Age", int(data['CustomerAge'].min()), int(data['CustomerAge'].max()), step=1)
tenure = st.slider("Tenure (months)", int(data['Tenure'].min()), int(data['Tenure'].max()), step=1)
service_usage = st.slider("Service Usage Rate", int(data['ServiceUsageRate'].min()), int(data['ServiceUsageRate'].max()), step=1)
support_calls = st.slider("Support Calls", int(data['SupportCalls'].min()), int(data['SupportCalls'].max()), step=1)
billing_delay = st.slider("Billing Delay Incidents", int(data['BillingDelay'].min()), int(data['BillingDelay'].max()), step=1)
total_expenditure = st.slider("Total Expenditure ($)", int(data['TotalExpenditure'].min()), int(data['TotalExpenditure'].max()), step=10)
recent_activity = st.slider("Recent Activity", int(data['RecentActivity'].min()), int(data['RecentActivity'].max()), step=1)

# Dropdowns for categorical features
plan_type = st.selectbox("Plan Type", data['PlanType'].unique())
agreement_duration = st.selectbox("Agreement Duration", data['AgreementDuration'].unique())

# Prepare input for prediction
input_features = pd.DataFrame({
    'CustomerAge': [age],
    'Tenure': [tenure],
    'ServiceUsageRate': [service_usage],
    'SupportCalls': [support_calls],
    'BillingDelay': [billing_delay],
    'TotalExpenditure': [total_expenditure],
    'RecentActivity': [recent_activity],
    'PlanType': [plan_type],
    'AgreementDuration': [agreement_duration]
})

# Use the model pipeline's preprocessor to transform input features
preprocessor = model.named_steps['preprocessor']
transformed_features = preprocessor.transform(input_features)

# Predict churn probability
classifier = model.named_steps['classifier']
churn_prob = classifier.predict_proba(transformed_features)[0][1]

# Display the result
st.write(f"Probability of Churn: {churn_prob * 100:.2f}%")
