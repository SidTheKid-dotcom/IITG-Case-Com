import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the trained model
model_path = "output/models/churn_model_enhanced.pkl"  # Update path if different
model = joblib.load(model_path)

# Step 2: Load your dataset (ensure it matches training preprocessing)
dataset_path = "input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv"  # Update path if different
data = pd.read_csv(dataset_path)

# Ensure the dataset matches the model's preprocessing pipeline
X = data[['CustomerAge', 'Tenure', 'ServiceUsageRate', 'SupportCalls', 
          'BillingDelay', 'TotalExpenditure', 'PlanType', 'AgreementDuration', 'RecentActivity']]

# Apply preprocessing (model contains pipeline)
preprocessed_X = model.named_steps['preprocessor'].transform(X)

# Step 3: Create a SHAP Explainer
explainer = shap.TreeExplainer(model.named_steps['classifier'])

# Step 4: Calculate SHAP values
shap_values = explainer.shap_values(preprocessed_X)

# Step 5: Visualize SHAP values
# Use the transformed data (preprocessed_X)
shap.summary_plot(shap_values, preprocessed_X, feature_names=X.columns)

# Interaction Plot
# Ensure the feature names align with the preprocessed data
shap.dependence_plot('Tenure', shap_values[1], preprocessed_X, interaction_index='PlanType', feature_names=X.columns)

# Note: Adjust feature names as per dataset headers
