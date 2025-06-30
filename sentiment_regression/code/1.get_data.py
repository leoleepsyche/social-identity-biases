from datasets import load_dataset
import json
import jieba
from sklearn.preprocessing import StandardScaler
import pandas as pd
import string
import re
import os
from tqdm import tqdm

# Load dataset and define dataset name for file naming
DATASET_NAME = "WildChat-1M"
ds = load_dataset("allenai/WildChat-1M")

# Filter out all Chinese data
print("Filtering Chinese data...")
chinese_ds = ds.filter(lambda example: example["language"] == "Chinese")

# Define in-group and out-group linguistic patterns
in_group_patterns = [
    "我们是", "我们的是", "我们通常", "我们的方式是",
    "我们经常", "我们相信"
]
out_group_patterns = [
    "他们是", "他们的是", "他们通常", "他们的方式是",
    "他们经常", "他们相信"
]


def extract_sentences_with_pattern(text, pattern):
    sentence_delimiters = r'[。！？；…~～——][”」』"]?|[\n\r]'
    sentences = re.split(sentence_delimiters, text)
    
    matched_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence and pattern in sentence:
            start_pos = text.find(sentence)
            if start_pos != -1:
                end_pos = start_pos + len(sentence)
                while end_pos < len(text) and text[end_pos] in '。！？':
                    end_pos += 1
                complete_sentence = text[start_pos:end_pos].strip()
                if complete_sentence:
                    matched_sentences.append(complete_sentence)
    
    return matched_sentences


# Store matched sentences
in_group_sentences = []
out_group_sentences = []

# Iterate over all conversations to extract matched sentences with progress bar
print("Extracting sentences with patterns...")
for example in tqdm(chinese_ds['train'], desc="Processing conversations"):
    
    for utterance in example["conversation"]:
        if "content" in utterance and isinstance(utterance["content"], str):
            content = utterance["content"]
            role = utterance.get("role", "unknown")
            # Extract model information if available
            model = example.get("model", "unknown")

            # Check for in-group patterns
            for pattern in in_group_patterns:
                matched_sentences = extract_sentences_with_pattern(content, pattern)
                for sentence in matched_sentences:
                    in_group_sentences.append({
                        "text": sentence,
                        "source": "we",
                        "role": role,
                        "marker": pattern,
                        "model": model,
                    })
            
            # Check for out-group patterns
            for pattern in out_group_patterns:
                matched_sentences = extract_sentences_with_pattern(content, pattern)
                for sentence in matched_sentences:
                    out_group_sentences.append({
                        "text": sentence,
                        "source": "they",
                        "role": role,
                        "marker": pattern,
                        "model": model,
                    })


print(f"Dataset: {DATASET_NAME}")
print(f"Size of Chinese dataset:{len(chinese_ds['train'])}")
print(f"Number of in-group sentences found: {len(in_group_sentences)}")
print(f"Number of out-group sentences found: {len(out_group_sentences)}")

# Create the data directory
os.makedirs("../data", exist_ok=True)

print("Saving extracted sentences to JSON files...")
# Save results to files with dataset name suffix
with open(f"../data/in_group_sentences_{DATASET_NAME}.json", "w", encoding="utf-8") as f:
    json.dump(in_group_sentences, f, ensure_ascii=False, indent=2)

with open(f"../data/out_group_sentences_{DATASET_NAME}.json", "w", encoding="utf-8") as f:
    json.dump(out_group_sentences, f, ensure_ascii=False, indent=2)


# Preprocess text before tokenization (for TTR and token counting only)
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

print("Calculating TTR and token counts...")
# Process in-group sentences
for sentence in tqdm(in_group_sentences, desc="Processing in-group sentences"):
    text = sentence["text"]
    ttr = calculate_ttr(text)
    total_tokens = count_tokens(text)

    all_sentences.append({
        "text": text,
        "source": sentence["source"],  # "we" or "they"
        "role": sentence["role"],
        "marker": sentence["marker"],
        "model": sentence["model"],  # Add model column
        "TTR": ttr,
        "total_tokens": total_tokens,
    })

# Process out-group sentences
for sentence in tqdm(out_group_sentences, desc="Processing out-group sentences"):
    text = sentence["text"]
    ttr = calculate_ttr(text)
    total_tokens = count_tokens(text)
    all_sentences.append({
        "text": text,
        "source": sentence["source"],  # "we" or "they"
        "role": sentence["role"],
        "marker": sentence["marker"],
        "model": sentence["model"],  # Add model column
        "TTR": ttr,
        "total_tokens": total_tokens,
    })

# Convert to DataFrame
print("Converting to DataFrame and scaling...")
df = pd.DataFrame(all_sentences)

# Standardize total_tokens
scaler = StandardScaler()
df['total_tokens_scaled'] = scaler.fit_transform(df[['total_tokens']])

print(f"Total number of sentences prepared for topic modeling and sentiment analysis: {len(df)}")

# Save with dataset name suffix
path = f"../result/1.group_data_{DATASET_NAME}.csv"
print(f"Saving final data to {path}...")
df.to_csv(path, index=False, encoding='utf-8')
