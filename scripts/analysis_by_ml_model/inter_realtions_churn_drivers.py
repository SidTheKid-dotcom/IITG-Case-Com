# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Load dataset
dataset_path = 'input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
data = pd.read_csv(dataset_path)

# Analyze categorical variables
categorical_columns = ['Sex', 'PlanType', 'AgreementDuration']

# Feature Engineering
data['BillingDelayRate'] = data['BillingDelay'] / data['Tenure']
data['ActivityRatio'] = data['RecentActivity'] / data['Tenure']
data['SpendingEfficiency'] = data['TotalExpenditure'] / data['Tenure']

# Split dataset into features and target
X = data.drop(columns=['Churn', 'UserID'])
y = data['Churn']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Load the pre-trained Random Forest model
model_path = 'output/models/churn_model.pkl'
with open(model_path, 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Make predictions
y_pred = rf_model.predict(X)
y_prob = rf_model.predict_proba(X)[:, 1]

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Classification Report:\n", classification_report(y, y_pred))
print("Accuracy Score:", accuracy_score(y, y_pred))
print("ROC AUC Score:", roc_auc_score(y, y_prob))

# Explore interdependencies
interdependencies = [
    ('BillingDelay', 'SupportCalls'),
    ('Tenure', 'ServiceUsageRate'),
    ('TotalExpenditure', 'RecentActivity'),
    ('BillingDelayRate', 'ActivityRatio'),
    ('Tenure', 'BillingDelay'),
    ('SupportCalls', 'ServiceUsageRate'),
    ('TotalExpenditure', 'Tenure'),
    ('BillingDelay', 'RecentActivity'),
    ('ActivityRatio', 'SpendingEfficiency'),
    ('ServiceUsageRate', 'RecentActivity')
]

for feature_x, feature_y in interdependencies:
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=feature_x, y=feature_y, hue='Churn', palette='coolwarm', alpha=0.7)
    plt.title(f"{feature_x} vs {feature_y} by Churn")
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.legend(title='Churn', loc='best')
    plt.show()

# Explore 3-way interdependency: Tenure vs BillingDelay vs SupportCalls
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='Tenure', y='BillingDelay', size='SupportCalls', hue='Churn', palette='coolwarm', alpha=0.7, sizes=(40, 400))
plt.title("Tenure vs BillingDelay vs SupportCalls by Churn")
plt.xlabel("Tenure")
plt.ylabel("BillingDelay")
plt.legend(title="Churn", loc='best')
plt.show()
