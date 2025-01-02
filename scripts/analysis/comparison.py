import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '../../input/Strategy Storm 2025 - Round 2 dataset - SSDataset.csv.csv'
data = pd.read_csv(file_path)

# Create a comprehensive comparison using multiple visualization types
metrics = ['CustomerAge', 'Tenure', 'ServiceUsageRate', 'SupportCalls',
          'BillingDelay', 'TotalExpenditure', 'RecentActivity']

# Calculate summary statistics
summary_stats = pd.DataFrame()
for metric in metrics:
    churned_mean = data[data['Churn'] == 1][metric].mean()
    not_churned_mean = data[data['Churn'] == 0][metric].mean()
    summary_stats = pd.concat([summary_stats, pd.DataFrame({
        'Metric': [metric],
        'Churned': [churned_mean],
        'Not_Churned': [not_churned_mean],
        'Difference': [churned_mean - not_churned_mean]
    })])

# Create a combination plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Bar plot showing absolute values
summary_stats_melted = summary_stats.melt(
    id_vars='Metric', 
    value_vars=['Churned', 'Not_Churned'],
    var_name='Group',
    value_name='Value'
)

sns.barplot(
    data=summary_stats_melted,
    x='Metric',
    y='Value',
    hue='Group',
    ax=ax1,
    palette=['#e74c3c', '#3498db']
)
ax1.set_title('Average Values by Churn Status')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

# Percentage difference plot
sns.barplot(
    data=summary_stats,
    x='Metric',
    y='Difference',
    ax=ax2,
    color='#2ecc71'
)
ax2.set_title('Difference Between Churned and Non-Churned Customers')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
ax2.axhline(y=0, color='black', linestyle='-', alpha=0.2)

plt.tight_layout()
plt.show()

# 3. Create a heatmap of correlations with churn
correlation_matrix = data[metrics + ['Churn']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix[['Churn']].sort_values(by='Churn', ascending=False),
    annot=True,
    cmap='RdBu',
    center=0,
    vmin=-1,
    vmax=1
)
plt.title('Correlation with Churn')
plt.tight_layout()
plt.show()