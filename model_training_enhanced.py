import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Step 1: Load the Dataset
data = pd.read_csv("./input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv")

# Step 2: Data Preprocessing
# Handle missing values (if any)
data = data.dropna()

# Separate features and target
X = data[['CustomerAge', 'Tenure', 'ServiceUsageRate', 'SupportCalls', 
          'BillingDelay', 'TotalExpenditure', 'PlanType', 'AgreementDuration', 'RecentActivity']]
y = data['Churn']

# Define categorical and numerical features
categorical_features = ['PlanType', 'AgreementDuration']
numerical_features = ['CustomerAge', 'Tenure', 'ServiceUsageRate', 'SupportCalls', 
                      'BillingDelay', 'TotalExpenditure', 'RecentActivity']

# Create preprocessors
categorical_transformer = OneHotEncoder(handle_unknown='ignore')
numerical_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Step 3: Build the Model Pipeline
model = RandomForestClassifier(random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# Step 4: Hyperparameter Tuning
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4],
    'classifier__max_features': ['auto', 'sqrt']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# Step 5: Train-Test Split and Cross-Validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the best model
best_model.fit(X_train, y_train)

# Step 6: Model Evaluation
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_proba)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
print(f"AUC-ROC Score: {auc_score:.2f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Step 7: Feature Importance
classifier = best_model.named_steps['classifier']
feature_names = (numerical_features +
                 list(grid_search.best_estimator_.named_steps['preprocessor'].transformers_[1][1].get_feature_names_out()))
importances = classifier.feature_importances_
sorted_indices = np.argsort(importances)[::-1]

print("\nFeature Importances:")
for i in sorted_indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# Step 8: Save the Model
joblib.dump(best_model, "./output/models/churn_model_enhanced.pkl")
print("Enhanced model saved as 'churn_model_enhanced.pkl'")
