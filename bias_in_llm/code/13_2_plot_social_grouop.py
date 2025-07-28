import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = pd.read_csv('.\data\logistic_result_social_group.csv')
data['Coefficient'] = np.exp(data['Coefficient']) # coefficient to odds ratio
data = data[data['SentimentClassifier']=='aliyun']
df1 = data[data['Type']=='in']
df2 = data[data['Type']=='out']
# social group name
social_grouops1 = df1['SocialGroup'].to_list()
social_grouops2 = df2['SocialGroup'].to_list()
social_groups = list(set(social_grouops1) & set(social_grouops2))

df1 = df1[df1['SocialGroup'].isin(social_groups)]
df2 = df2[df2['SocialGroup'].isin(social_groups)]
# average coefficient of every social group
coef1_means = df1['Coefficient'].to_list()
coef2_means = df2['Coefficient'].to_list()
# std of every social group
coef1_stds = df1['Std'].to_list()
coef2_stds = df2['Std'].to_list()


y = np.arange(len(social_groups))
offset = 0.1  # shift of coef1 and coef2

plt.figure(figsize=(8, 6))

# plot coef1
plt.errorbar(coef1_means, y - offset, xerr=coef1_stds, fmt='o', capsize=5, label='Ingroup Solidarity', color='blue')

# plot coef2
plt.errorbar(coef2_means, y + offset, xerr=coef2_stds, fmt='o', capsize=5, label='Outgroup Hostility', color='orange')

# set label of yaxis
plt.yticks(y, social_groups)
plt.xlabel('Odds Ratio')
plt.ylabel('Social Group')
plt.title('Regression Odds Ratio (Mean ± Std)')
# plt.grid(True, linestyle='--', alpha=0.6)
plt.axvline(x=1, color='gray', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('.\\data\\logistic_result_social_group.pdf', format='pdf')