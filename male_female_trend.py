# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

# Load dataset
dataset_path = './input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
data = pd.read_csv(dataset_path)

# Set the style of the plot
sns.set(style="whitegrid")

# Create a boxplot to visualize tenure for churned customers
plt.figure(figsize=(8, 6))
sns.boxplot(x=data['Tenure'], color="skyblue")
plt.title("Distribution of Tenure for Churned Customers", fontsize=14)
plt.xlabel("Tenure", fontsize=12)
plt.show()

# Filter data where Churn = 0
not_churned_data = data[data['Churn'] == 0]

# Create a boxplot to visualize tenure for customers who did not churn
plt.figure(figsize=(8, 6))
sns.boxplot(x=not_churned_data['Tenure'], color="red")
plt.title("Distribution of Tenure for Non-Churned Customers", fontsize=14)
plt.xlabel("Tenure", fontsize=12)
plt.show()

# Import necessary library
from scipy import stats

# Filter data for churned and non-churned customers
churned_data = data[data['Churn'] == 1]
not_churned_data = data[data['Churn'] == 0]

# Calculate statistical data for churned customers
average_churned = churned_data['Tenure'].mean()
median_churned = churned_data['Tenure'].median()
quartiles_churned = churned_data['Tenure'].quantile([0.25, 0.5, 0.75])
mode_churned = stats.mode(churned_data['Tenure'], keepdims=True).mode[0]
std_dev_churned = churned_data['Tenure'].std()

# Calculate statistical data for non-churned customers
average_not_churned = not_churned_data['Tenure'].mean()
median_not_churned = not_churned_data['Tenure'].median()
quartiles_not_churned = not_churned_data['Tenure'].quantile([0.25, 0.5, 0.75])
mode_not_churned = stats.mode(not_churned_data['Tenure'], keepdims=True).mode[0]
std_dev_not_churned = not_churned_data['Tenure'].std()

# Print results for churned customers
print("Churned Customers (Churn = 1):")
print("Average Tenure:", average_churned)
print("Median Tenure:", median_churned)
print("Quartiles Tenure:", quartiles_churned.to_dict())
print("Mode Tenure:", mode_churned)
print("Standard Deviation:", std_dev_churned)

# Print results for non-churned customers
print("\nNon-Churned Customers (Churn = 0):")
print("Average Tenure:", average_not_churned)
print("Median Tenure:", median_not_churned)
print("Quartiles Tenure:", quartiles_not_churned.to_dict())
print("Mode Tenure:", mode_not_churned)
print("Standard Deviation:", std_dev_not_churned)

# Analyze categorical and numerical variables
categorical_columns = ['Sex', 'PlanType', 'AgreementDuration']
numerical_columns = ['Tenure', 'TotalExpenditure', 'BillingDelay', 'RecentActivity']

# Feature Engineering
data['BillingDelayRate'] = data['BillingDelay'] / data['Tenure']
data['ActivityRatio'] = data['RecentActivity'] / data['Tenure']
data['SpendingEfficiency'] = data['TotalExpenditure'] / data['Tenure']

# Split dataset into features and target
X = data.drop(columns=['Churn', 'UserID'])
y = data['Churn']

# Convert categorical variables into dummy/indicator variables
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

# Load the existing Random Forest model
model_path = './output/models/churn_model.pkl'
rf_model = joblib.load(model_path)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Make predictions and evaluate the model
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_prob))

# Feature Importance
if hasattr(rf_model, 'feature_importances_'):
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print("\nFeature Importances:")
    print(feature_importances)

# Statistical Analysis
churned = data[data['Churn'] == 1]
not_churned = data[data['Churn'] == 0]

print("\nT-Test Results for Numerical Features:")
for col in numerical_columns:
    t_stat, p_val = ttest_ind(churned[col], not_churned[col], nan_policy='omit')
    print(f"{col}: t-statistic = {t_stat:.2f}, p-value = {p_val:.4f}")

print("\nChi-Square Test Results for Categorical Features:")
for col in categorical_columns:
    contingency_table = pd.crosstab(data[col], data['Churn'])
    chi2, p_val, dof, _ = chi2_contingency(contingency_table)
    print(f"{col}: chi2 = {chi2:.2f}, p-value = {p_val:.4f}")

# K-Means Clustering
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(pca_data)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='Set1', style=y)
plt.title("Customer Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.show()

# Export Feature Importances and Statistical Results
if hasattr(rf_model, 'feature_importances_'):
    drivers = feature_importances.copy()
    drivers['Statistical_Significance'] = [
        ttest_ind(churned[feature], not_churned[feature], nan_policy='omit')[1]
        if feature in numerical_columns else None
        for feature in drivers['Feature']
    ]
    drivers.to_csv('churn_drivers.csv', index=False)
    print("Churn drivers exported to 'churn_drivers.csv'")

# Calculate churn rates by gender
gender_churn = data.groupby('Sex')['Churn'].mean() * 100  # Convert to percentage
print("Churn Rates by Gender (%):")
print(gender_churn)

# Visualize churn rates by gender
plt.figure(figsize=(8, 6))
sns.barplot(x=gender_churn.index, y=gender_churn.values, palette='Set1')
plt.title("Churn Rates by Gender")
plt.xlabel("Gender")
plt.ylabel("Churn Rate (%)")
plt.show()

# Perform Chi-Square Test for Gender
contingency_table = pd.crosstab(data['Sex'], data['Churn'])
chi2, p_val, dof, expected = chi2_contingency(contingency_table)

print("\nChi-Square Test for Gender:")
print(f"Chi-Square Statistic: {chi2:.2f}")
print(f"P-Value: {p_val:.4f}")
print(f"Degrees of Freedom: {dof}")

# Segment data by gender
female_data = data[data['Sex'] == 'Female']
male_data = data[data['Sex'] == 'Male']

# T-Test for numerical features for females
print("\nT-Test Results for Numerical Features (Females):")
for col in numerical_columns:
    t_stat, p_val = ttest_ind(female_data[col], female_data[female_data['Churn'] == 0][col], nan_policy='omit')
    print(f"{col}: t-statistic = {t_stat:.2f}, p-value = {p_val:.4f}")

# T-Test for numerical features for males
print("\nT-Test Results for Numerical Features (Males):")
for col in numerical_columns:
    t_stat, p_val = ttest_ind(male_data[col], male_data[male_data['Churn'] == 0][col], nan_policy='omit')
    print(f"{col}: t-statistic = {t_stat:.2f}, p-value = {p_val:.4f}")

# Chi-Square Test for categorical features for females
print("\nChi-Square Test Results for Categorical Features (Females):")
for col in categorical_columns:
    contingency_table = pd.crosstab(female_data[col], female_data['Churn'])
    chi2, p_val, dof, _ = chi2_contingency(contingency_table)
    print(f"{col}: chi2 = {chi2:.2f}, p-value = {p_val:.4f}")

# Chi-Square Test for categorical features for males
print("\nChi-Square Test Results for Categorical Features (Males):")
for col in categorical_columns:
    contingency_table = pd.crosstab(male_data[col], male_data['Churn'])
    chi2, p_val, dof, _ = chi2_contingency(contingency_table)
    print(f"{col}: chi2 = {chi2:.2f}, p-value = {p_val:.4f}")

# 1. Distribution of Numerical Features by Gender and Churn
for col in numerical_columns:
    plt.figure(figsize=(12, 5))
    
    # Create side-by-side plots for female and male data
    plt.subplot(1, 2, 1)
    sns.boxplot(x='Churn', y=col, data=female_data, hue='Churn', legend=False)
    plt.title(f"Female: {col} vs Churn")
    plt.xlabel("Churn (0=No, 1=Yes)")
    
    plt.subplot(1, 2, 2)
    sns.boxplot(x='Churn', y=col, data=male_data, hue='Churn', legend=False)
    plt.title(f"Male: {col} vs Churn")
    plt.xlabel("Churn (0=No, 1=Yes)")
    
    plt.tight_layout()
    plt.show()

# 2. Churn Rate by Categorical Features for Females
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, 3, i)
    churn_rate = female_data.groupby(col)['Churn'].mean() * 100
    sns.barplot(x=churn_rate.index, y=churn_rate.values, hue=churn_rate.index, legend=False)
    plt.title(f"Female: Churn Rate by {col}")
    plt.xlabel(col)
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Churn Rate by Categorical Features for Males
plt.figure(figsize=(15, 5))
for i, col in enumerate(categorical_columns, 1):
    plt.subplot(1, 3, i)
    churn_rate = male_data.groupby(col)['Churn'].mean() * 100
    sns.barplot(x=churn_rate.index, y=churn_rate.values, hue=churn_rate.index, legend=False, palette='Set1')
    plt.title(f"Male: Churn Rate by {col}")
    plt.xlabel(col)
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

""" # Assuming 'Age' and 'SupportCalls' are columns in your dataset
# Create a scatter plot for Billing Delay vs Age of Customer, colored by Sex of Customer
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Age', y='BillingDelay', hue='Sex', data=data, palette='Set1')
plt.title('Billing Delay vs Age of Customer')
plt.xlabel('Age of Customer')
plt.ylabel('Billing Delay')
plt.legend(title='Sex')
plt.show()

# Create a box plot for Support Calls by Sex of Customer
plt.figure(figsize=(8, 6))
sns.boxplot(x='Sex', y='SupportCalls', data=data, palette='Set1')
plt.title('Support Calls by Sex of Customer')
plt.xlabel('Sex of Customer')
plt.ylabel('Number of Support Calls')
plt.show()
 """