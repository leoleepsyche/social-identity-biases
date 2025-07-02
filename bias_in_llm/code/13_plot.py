import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('.\\data\\logistic_result.csv')  # Replace with your actual file path or use pd.read_clipboard() to paste table
data = data[data['type'] == 'out']

# Construct matrix: rows are LLMs, columns are SentimentClassifiers, values are Coefficients
coef_matrix = data.pivot(index='LLM', columns='SentimentClassifier', values='Coefficient')

def format_pvalue(p):
    if p < 1e-4:
        return 'p<1e-4'
    else:
        return f'{p:.1e}'

data['Label'] = data.apply(lambda row: f"{row['Coefficient']:.2f}\n({format_pvalue(row['PValue'])})", axis=1)
label_matrix = data.pivot(index='LLM', columns='SentimentClassifier', values='Label')

# # Construct annotation label matrix: format as Coefficient (PValue)
# data['Label'] = data.apply(lambda row: f"{row['Coefficient']:.2f}\n({row['PValue']:.1e})", axis=1)
# label_matrix = data.pivot(index='LLM', columns='SentimentClassifier', values='Label')

# Plot heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(coef_matrix, annot=label_matrix, fmt='', cmap='Blues', center=0, linewidths=0.5, cbar_kws={'label': 'Coefficient'})
plt.title('Coefficient Heatmap with p-Value')
plt.tight_layout()
plt.show()
plt.savefig('.\\data\\logistic_result.pdf', format='pdf')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('.\\data\\logistic_result.csv')
data = data[data['SentimentClassifier'] == 'erlangshen']
df1 = data[data['type'] == 'in']
df2 = data[data['type'] == 'out']

# Model names
models = df1['LLM'].to_list()

# Mean of two coefficients for each model
coef1_means = df1['Coefficient'].to_list()
coef2_means = df2['Coefficient'].to_list()

# Standard deviation of each coefficient
coef1_stds = df1['Std'].to_list()
coef2_stds = df2['Std'].to_list()

# Y-axis positions
y = np.arange(len(models))
offset = 0.1  # Vertical offset between coefficient 1 and coefficient 2

plt.figure(figsize=(8, 6))

# Plot coefficient 1
plt.errorbar(coef1_means, y - offset, xerr=coef1_stds, fmt='o', capsize=5, label='Ingroup solidarity', color='blue')

# Plot coefficient 2
plt.errorbar(coef2_means, y + offset, xerr=coef2_stds, fmt='o', capsize=5, label='Outgroup hostility', color='orange')

# Set y-axis labels
plt.yticks(y, models)
plt.xlabel('Odds Ratio')
plt.ylabel('Model')
plt.title('Regression Coefficients (Mean ± Std)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
