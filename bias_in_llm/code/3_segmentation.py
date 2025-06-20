import pandas as pd
import jieba
import re

# Stopword list, should be prepared with one word per line
with open("hit_stopwords.txt", "r", encoding='utf-8') as f:
    stopwords = set([line.strip() for line in f])

# Keep only Chinese characters
def keep_chinese(text):
    return ''.join(re.findall(r'[\u4e00-\u9fa5]', text))

# Cleaning + tokenization + stopword removal
def process_sentence(sentence, stopwords):
    # Keep only Chinese characters
    sentence = keep_chinese(sentence)
    # Tokenize the sentence
    words = jieba.lcut(sentence)
    # Remove stopwords
    words = [w for w in words if w not in stopwords and w.strip()]
    return words

# Read data
df = pd.read_csv("./data/all_data.csv", encoding='utf-8-sig')

# Process each sentence
processed = df['sentence'].apply(lambda x: process_sentence(x, stopwords))

# Join tokens into a single string with spaces
processed = processed.apply(lambda x: " ".join(x))

# Add to DataFrame as a new column
df['sentence_segmented'] = processed

# Check the result
print("Sample of cleaned data:")
print(df.head())    

# Save to a new file
df.to_csv("./data/all_data.csv", index=False, encoding='utf-8-sig')
