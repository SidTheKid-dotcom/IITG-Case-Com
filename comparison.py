import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = './input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
data = pd.read_csv(file_path)

# Split the data into churned and non-churned customers
churned = data[data['Churn'] == 1]
not_churned = data[data['Churn'] == 0]

# Descriptive statistics for both groups
churned_stats = churned.describe()
not_churned_stats = not_churned.describe()

# Visualize differences in key metrics (CustomerAge, Tenure, TotalExpenditure)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
# Visualize differences in key metrics using line plots

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# CustomerAge line plot
sns.lineplot(churned['CustomerAge'], color='red', label='Churned', ax=axes[0])
sns.lineplot(not_churned['CustomerAge'], color='blue', label='Not Churned', ax=axes[0])
axes[0].set_title('Customer Age Distribution')
axes[0].set_xlabel('Age')
axes[0].legend()

# Tenure line plot
sns.lineplot(churned['Tenure'], color='red', label='Churned', ax=axes[1])
sns.lineplot(not_churned['Tenure'], color='blue', label='Not Churned', ax=axes[1])
axes[1].set_title('Tenure Distribution')
axes[1].set_xlabel('Tenure (Months)')
axes[1].legend()

# TotalExpenditure line plot
sns.lineplot(churned['TotalExpenditure'], color='red', label='Churned', ax=axes[2])
sns.lineplot(not_churned['TotalExpenditure'], color='blue', label='Not Churned', ax=axes[2])
axes[2].set_title('Total Expenditure Distribution')
axes[2].set_xlabel('Total Expenditure ($)')
axes[2].legend()

plt.tight_layout()
plt.show()

# Plotting average distributions for better visualization of group comparisons

# Calculate average values for churned and non-churned groups
avg_values = pd.DataFrame({
    'Metric': ['CustomerAge', 'Tenure', 'ServiceUsageRate', 'SupportCalls', 
               'BillingDelay', 'TotalExpenditure', 'RecentActivity'],
    'Churned': [
        churned['CustomerAge'].mean(),
        churned['Tenure'].mean(),
        churned['ServiceUsageRate'].mean(),
        churned['SupportCalls'].mean(),
        churned['BillingDelay'].mean(),
        churned['TotalExpenditure'].mean(),
        churned['RecentActivity'].mean()
    ],
    'Not Churned': [
        not_churned['CustomerAge'].mean(),
        not_churned['Tenure'].mean(),
        not_churned['ServiceUsageRate'].mean(),
        not_churned['SupportCalls'].mean(),
        not_churned['BillingDelay'].mean(),
        not_churned['TotalExpenditure'].mean(),
        not_churned['RecentActivity'].mean()
    ]
})

# Melt the data for easy plotting
avg_values_melted = avg_values.melt(id_vars="Metric", var_name="Group", value_name="Average")

# Line plot for average comparison
plt.figure(figsize=(12, 6))
sns.lineplot(data=avg_values_melted, x="Metric", y="Average", hue="Group", marker="o", palette="viridis")
plt.title("Average Comparison Between Churned and Not Churned Customers")
plt.xlabel("Metric")
plt.ylabel("Average Value")
plt.xticks(rotation=45)
plt.legend(title="Group")
plt.tight_layout()
plt.show()

