import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# Step 1: Load the trained model
model_path = "output/models/churn_model_enhanced.pkl"  # Update path if different
model = joblib.load(model_path)

# Step 2: Load the test dataset
dataset_path = "input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv"  # Update path if different
data = pd.read_csv(dataset_path)

# Separate features and target
X = data[['CustomerAge', 'Tenure', 'ServiceUsageRate', 'SupportCalls', 
          'BillingDelay', 'TotalExpenditure', 'PlanType', 'AgreementDuration', 'RecentActivity']]
y = data['Churn']

# Split data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocess the test data
preprocessed_X_test = model.named_steps['preprocessor'].transform(X_test)

# Step 3: Evaluate the model
# Predict labels
y_pred = model.predict(X_test)
# Predict probabilities (for ROC-AUC)
y_proba = model.predict_proba(X_test)[:, 1]

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.named_steps['classifier'].classes_)
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ROC-AUC Score
auc_score = roc_auc_score(y_test, y_proba)
print(f"\nROC-AUC Score: {auc_score:.2f}")
