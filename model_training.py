import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_and_save_model(data_path, model_path):
    # Load dataset
    data = pd.read_csv(data_path)

    # Feature Engineering
    categorical_columns = ['Sex', 'PlanType', 'AgreementDuration']
    data['BillingDelayRate'] = data['BillingDelay'] / data['Tenure']
    data['ActivityRatio'] = data['RecentActivity'] / data['Tenure']
    data['SpendingEfficiency'] = data['TotalExpenditure'] / data['Tenure']

    # Split dataset into features and target
    X = data.drop(columns=['Churn', 'UserID'])
    y = data['Churn']

    # Convert categorical variables into dummy/indicator variables
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Split data into training and testing sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a Random Forest model
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the trained model
    with open(model_path, 'wb') as file:
        pickle.dump(rf_model, file)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    data_path = './input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
    model_path = './output/models/churn_model.pkl'
    train_and_save_model(data_path, model_path)
