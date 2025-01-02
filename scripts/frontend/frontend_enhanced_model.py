import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# Add custom CSS to adjust layout
st.markdown("""
    <style>
        .block-container {
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 100%;
        }
        [data-testid="column"] {
            padding: 0 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
model = joblib.load("../../output/models/churn_model_enhanced.pkl")

# Load dataset for dynamic ranges
dataset_path = '../../input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
data = pd.read_csv(dataset_path)

st.title("Customer Churn Prediction and Analysis")

# Create container with custom width
with st.container():
    # Create two columns with padding
    col1, col2 = st.columns([1, 2], gap="large")  # Added gap parameter
    
    # Input sliders and dropdowns in the left column
    with col1:
        st.header("Input Parameters")
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
        

    # Graph in the right column
    with col2:
        st.header("Impact of Parameters on Churn")
        
        # Display the churn probability
        st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")

        # Select parameter to analyze
        parameter_to_vary = st.selectbox(
            "Select a parameter to analyze its impact on churn probability:",
            ['CustomerAge', 'Tenure', 'ServiceUsageRate', 'SupportCalls', 'BillingDelay', 'TotalExpenditure', 'RecentActivity']
        )
        
        # Generate data for the graph
        varied_range = np.linspace(
            data[parameter_to_vary].min(),
            data[parameter_to_vary].max(),
            num=100
        )
        churn_probs = []
        
        # Loop through varied values while keeping others constant
        for value in varied_range:
            temp_features = input_features.copy()
            temp_features[parameter_to_vary] = value
            transformed_temp = preprocessor.transform(temp_features)
            churn_prob = classifier.predict_proba(transformed_temp)[0][1]
            churn_probs.append(churn_prob)
        
        # Create a DataFrame for visualization
        graph_data = pd.DataFrame({
            parameter_to_vary: varied_range,
            "Churn Probability": churn_probs
        })
        
        # Plot the graph
        fig = px.line(
            graph_data,
            x=parameter_to_vary,
            y="Churn Probability",
            title=f"Impact of {parameter_to_vary} on Churn Probability",
            labels={"x": parameter_to_vary, "y": "Churn Probability"},
            template="plotly_white"
        )
        
        # Update layout to remove margins
        fig.update_layout(
            margin=dict(l=0, r=0, t=40, b=0),
            width=None,  # Allow responsive width
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Highlight inflection points
        threshold = 0.5  # Example: 50% churn probability
        inflection_points = graph_data[graph_data["Churn Probability"] >= threshold]
        if not inflection_points.empty:
            st.write(f"**Inflection Point Detected**: {parameter_to_vary} value where churn exceeds {threshold * 100:.0f}% is approximately:")
            st.write(f"{inflection_points.iloc[0][parameter_to_vary]:.2f}")
        else:
            st.write(f"No inflection point detected for {threshold * 100:.0f}% churn probability.")