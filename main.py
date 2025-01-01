import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
file_path = './input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
dataset = pd.read_csv(file_path)

# Display basic information about the dataset
dataset_info = dataset.info()
dataset_head = dataset.head()

# Check for duplicate entries
duplicates = dataset.duplicated().sum()

# Inspect value distributions of key numeric attributes
numeric_columns = ['CustomerAge', 'Tenure', 'ServiceUsageRate', 
                   'SupportCalls', 'BillingDelay', 'TotalExpenditure', 'RecentActivity']
numeric_summary = dataset[numeric_columns].describe()

# Examine distribution of the target variable (Churn)
churn_distribution = dataset['Churn'].value_counts(normalize=True) * 100

import matplotlib.pyplot as plt
import seaborn as sns

# Set up visual style
sns.set(style="whitegrid")

# Visualize numeric attributes
fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle("Distribution of Numeric Attributes", fontsize=16)

numeric_columns = ['CustomerAge', 'Tenure', 'ServiceUsageRate', 
                   'SupportCalls', 'BillingDelay', 'TotalExpenditure', 'RecentActivity']
for ax, col in zip(axes.flatten(), numeric_columns):
    sns.histplot(data=dataset, x=col, kde=True, ax=ax, color='teal')
    ax.set_title(f"Distribution of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

# Hide any empty subplots
for ax in axes.flatten()[len(numeric_columns):]:
    ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Visualize correlations between numeric attributes and churn
correlation_data = dataset[numeric_columns + ['Churn']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Matrix of Numeric Attributes with Churn")
plt.show()
