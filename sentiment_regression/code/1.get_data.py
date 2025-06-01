from datasets import load_dataset
import json
import jieba
from sklearn.preprocessing import StandardScaler
import pandas as pd
import string
import re
import os

ds = load_dataset("allenai/WildChat")
# Filter out all Chinese data
chinese_ds = ds.filter(lambda example: example["language"] == "Chinese")

# Define in-group and out-group linguistic patterns
in_group_patterns = [
    "我们是", "我们的是", "我们通常", "我们的方式是",
    "我们经常", "我们相信", "我们认为", "我们觉得"
]
out_group_patterns = [
    "他们是", "他们的是", "他们通常", "他们的方式是",
    "他们经常", "他们相信", "他们认为", "他们觉得"
]


# Store matched sentences
in_group_sentences = []
out_group_sentences = []

# Iterate over all conversations to extract matched sentences
for example in chinese_ds['train']:
    
    for utterance in example["conversation"]:
        if "content" in utterance and isinstance(utterance["content"], str):
            content = utterance["content"]
            role = utterance.get("role", "unknown")

            # Check for in-group patterns
            for pattern in in_group_patterns:
                if pattern in content:
                    in_group_sentences.append({
                        "text": content,
                        "source": "we",
                        "role": role,
                        "marker": pattern,
                    })
                    break
            
            # Check for out-group patterns
            for pattern in out_group_patterns:
                if pattern in content:
                    out_group_sentences.append({
                        "text": content,
                        "source": "they",
                        "role": role,
                        "marker": pattern,
                    })
                    break

# 输出结果
print(f"Size of Chinese dataset:{len(chinese_ds['train'])}")
print(f"Number of in-group sentences found: {len(in_group_sentences)}")
print(f"Number of out-group sentences found: {len(out_group_sentences)}")

# Create the data directory
os.makedirs("../data", exist_ok=True)

# Save results to files
with open("../data/in_group_sentences.json", "w", encoding="utf-8") as f:
    json.dump(in_group_sentences, f, ensure_ascii=False, indent=2)

with open("../data/out_group_sentences.json", "w", encoding="utf-8") as f:
    json.dump(out_group_sentences, f, ensure_ascii=False, indent=2)


# Preprocess text before tokenization
def preprocess_text_for_tokenization(text):
    text = text.lower()
    chinese_punctuation = '！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—''‛""„‰′″‴‵‶‷‸‹›※‼‽‾‿⁀⁁⁂⁃⁅⁆⁇⁈⁉⁊⁋⁌⁍⁎⁏⁐⁑⁒⁓⁔⁕⁖⁗⁘⁙⁚⁛⁜⁝⁞'

    # Remove punctuation and symbols
    for punct in string.punctuation + chinese_punctuation:
        text = text.replace(punct, ' ')

    # Remove numbers (consistent with R's remove_numbers = TRUE)
    clean_token = re.sub(r'\d+', '', text)

    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# Compute TTR (type-token ratio)
def calculate_ttr(text):
    processed_text = preprocess_text_for_tokenization(text)
    tokens = list(jieba.cut(processed_text))

    # Filter valid tokens: keep only Chinese characters or alphabetic words
    valid_tokens = []
    for token in tokens:
        token = token.strip()
        if token and (re.search(r'[\u4e00-\u9fff]', token) or token.isalpha()):
            valid_tokens.append(token)

    # Calculate TTR
    if len(valid_tokens) == 0:
        return 0.0

    unique_tokens = set(valid_tokens)
    ttr = len(unique_tokens) / len(valid_tokens)
    return ttr


# Count total number of tokens
def count_tokens(text):
    processed_text = preprocess_text_for_tokenization(text)
    tokens = list(jieba.cut(processed_text))

    # Filter valid tokens
    valid_tokens = []
    for token in tokens:
        token = token.strip()
        if token and (re.search(r'[\u4e00-\u9fff]', token) or token.isalpha()):
            valid_tokens.append(token)

    return len(valid_tokens)


# Prepare data for analysis
all_sentences = []

# Process in-group sentences
for sentence in in_group_sentences:
    text = sentence["text"]
    ttr = calculate_ttr(text)
    total_tokens = count_tokens(text)

    all_sentences.append({
        "text": text,
        "source": sentence["source"],  # "we" or "they"
        "role": sentence["role"],
        "marker": sentence["marker"],
        "TTR": ttr,
        "total_tokens": total_tokens,
    })

# Process out-group sentences
for sentence in out_group_sentences:
    text = sentence["text"]
    ttr = calculate_ttr(text)
    total_tokens = count_tokens(text)
    all_sentences.append({
        "text": text,
        "source": sentence["source"],  # "we" or "they"
        "role": sentence["role"],
        "marker": sentence["marker"],
        "TTR": ttr,
        "total_tokens": total_tokens,
    })

# Convert to DataFrame
df = pd.DataFrame(all_sentences)

# Standardize total_tokens
scaler = StandardScaler()
df['total_tokens_scaled'] = scaler.fit_transform(df[['total_tokens']])

print(f"Total number of sentences prepared for topic modeling and sentiment analysis: {len(df)}")

path = "../result/1.group_data.csv"
# Save the prepared data
df.to_csv(path, index=False, encoding='utf-8')

print(f"Data preparation completed and saved to {path}")
