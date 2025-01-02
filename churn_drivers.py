# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve

# Load dataset
dataset_path = './input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
data = pd.read_csv(dataset_path)

# Analyze categorical variables
categorical_columns = ['Sex', 'PlanType', 'AgreementDuration']

# Visualize distributions and churn relationships
plt.figure(figsize=(20, 6))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, 3, i)
    sns.countplot(data=data, x=col, hue='Churn', palette='Set2')
    plt.title(f"{col} vs Churn")
    plt.xlabel(col)
    plt.ylabel("Count")

plt.tight_layout()
plt.show()

# Feature Engineering
data['BillingDelayRate'] = data['BillingDelay'] / data['Tenure']
data['ActivityRatio'] = data['RecentActivity'] / data['Tenure']
data['SpendingEfficiency'] = data['TotalExpenditure'] / data['Tenure']

# Split dataset into features and target
X = data.drop(columns=['Churn', 'UserID'])
y = data['Churn']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\nTop 5 Feature Importances:")
print(feature_importances.head())

# Extract insights from top 5 features
top_features = feature_importances.head()['Feature'].values
for feature in top_features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='Churn', y=feature.split('_')[0], palette='Set2')
    plt.title(f"Churn vs {feature}")
    plt.xlabel("Churn")
    plt.ylabel(feature)
    plt.show()

# Plot Feature Importances
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')
plt.title('Feature Importances')
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
