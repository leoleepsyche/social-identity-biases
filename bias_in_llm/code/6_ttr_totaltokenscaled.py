import pandas as pd
import jieba
from sklearn.preprocessing import StandardScaler
import os

# Tokenization function + TTR (Type-Token Ratio)
def calc_ttr(text):
    tokens = list(jieba.cut(text))
    token_count = len(tokens)
    type_count = len(set(tokens))
    ttr = type_count / token_count if token_count > 0 else 0
    return token_count, ttr

# Read existing data
df = pd.read_csv(".\\data\\all_data_berttopic_stm.csv", encoding="utf-8-sig")

# Tokenization function + TTR (repeated, optional to keep only once)
def calc_ttr(text):
    tokens = list(jieba.cut(text))
    token_count = len(tokens)
    type_count = len(set(tokens))
    ttr = type_count / token_count if token_count > 0 else 0
    return token_count, ttr

# Calculate total number of tokens and TTR
token_counts = []
ttrs = []

for sentence in df['sentence']:
    tokens, ttr = calc_ttr(sentence)
    token_counts.append(tokens)
    ttrs.append(ttr)

df['TokenCount'] = token_counts
df['TTR'] = ttrs

# TotalTokenScaled: Normalize TokenCount (scaled between 0 and 1)
scaler = StandardScaler()
df['TotalTokenScaled'] = scaler.fit_transform(df[['TokenCount']])

# Save the results
df.to_csv(".\\data\\all_data_berttopic_stm_control.csv", index=False, encoding='utf-8-sig')

print("TTR and TotalTokenScaled calculation completed ✅")

