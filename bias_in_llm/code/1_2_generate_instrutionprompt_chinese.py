import os
import argparse
import pandas as pd
from typing import List
from nltk import ngrams
import re
from tqdm import tqdm
import pickle
import jieba
import time
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, help='how many sentences to generate', default=500)
args = parser.parse_args()
num_of_sentences = args.n



chinese_punctuations = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～「」、。，；：？！…—～"

def remove_punctuation(text: str) -> str:
    return re.sub(f"[{chinese_punctuations}]", "", text)

def split_first_sentence(text: str) -> str:
    match = re.search(r'(.+?[。！？])', text)
    return match.group(1) if match else text

def filter_sentences(sentence_candidates: List[str]) -> List[str]:
    valid_sentences = []
    all_valid_sentence_5grams = []
    for new_sentence in sentence_candidates:
        if len(new_sentence) < 10:
            continue
        new_sentence = re.sub('\n+', ' ', new_sentence)
        new_sentence = split_first_sentence(new_sentence)
        sentence_no_punct = remove_punctuation(new_sentence)
        words = list(jieba.cut(sentence_no_punct))
        if len(words) <= 5:
            continue
        new_5grams = set(ngrams(words, 5))
        if all(new_5grams.isdisjoint(old_5grams) for old_5grams in all_valid_sentence_5grams):
            all_valid_sentence_5grams.append(new_5grams)
            valid_sentences.append(re.sub(' +', ' ', new_sentence))
    return valid_sentences

group_prefixes = {
    "we": ["我们是", "我们通常", "我们经常", "我们相信"],
    "they": ["他们是", "他们通常", "他们经常", "他们相信"],
    "they_f": ["她们是", "她们通常", "她们经常", "她们相信"],
}

group_filenames = {
    "we": ['we_are', 'we_typically', 'we_often', 'we_believe'],
    "they": ['they_are', 'they_typically', 'they_often', 'they_believe'],
    "they_f": ['they_are_female', 'they_typically_female', 'they_often_female', 'they_believe_female'],
}

dataset = pd.read_csv('./data/random_sentences.csv')


base_dir = './data/'
save_directory = base_dir + 'deepseek-v3/'
os.makedirs(save_directory, exist_ok=True)

client = OpenAI(api_key="", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

def call_api(prompt: str) -> str:
    response = client.chat.completions.create(
        model="deepseek-v3",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        top_p=0.95
    )
    return response.choices[0].message.content

def generate_sentences(prefix, file_name):
    generated = set()
    count = 0
    start_time = time.time()
    print(f"\n[开始] 正在生成 {file_name} 的句子，以 “{prefix}” 开头，目标数量：{num_of_sentences}", flush=True)

    while count < num_of_sentences:
        chat_text = dataset['sentence'][count % len(dataset)] 
        prompt = f"上下文：{chat_text}\n现在生成一个以“{prefix}”开头的句子"

        try:
            text = call_api(prompt)
        except Exception as e:
            print(f"[错误] API 调用失败（第 {count} 个）: {e}", flush=True)
            continue

        text = split_first_sentence(re.sub('\n+', ' ', text))

        if text not in generated:
            generated.add(text)
            count += 1
            if count % 50 == 0 or count == num_of_sentences:
                elapsed = time.time() - start_time
                print(f"[进度] 已生成 {count}/{num_of_sentences} 条句子，用时 {elapsed:.1f} 秒", flush=True)

    print(f"[完成] 原始句子生成完毕，共计：{len(generated)} 条。开始过滤……", flush=True)

    generated = list(generated)
    filtered = filter_sentences(generated)

    print(f"[完成] 过滤后保留 {len(filtered)} 条句子，正在保存到文件...", flush=True)

    with open(f"{save_directory}{file_name}_sentences.pkl", 'wb') as f:
        pickle.dump(generated, f)
    pd.DataFrame(generated, columns=['sentence']).to_csv(
        f"{save_directory}{file_name}_sentences.csv", index=False, encoding='utf-8-sig'
    )

    with open(f"{save_directory}{file_name}_filtered_sentences.pkl", 'wb') as f:
        pickle.dump(filtered, f)
    pd.DataFrame(filtered, columns=['sentence']).to_csv(
        f"{save_directory}{file_name}_filtered_sentences.csv", index=False, encoding='utf-8-sig'
    )

    print(f"[保存完成] {file_name} 结果保存完毕。\n", flush=True)


for group in group_prefixes:
    for prefix, file_name in zip(group_prefixes[group], group_filenames[group]):
        generate_sentences(prefix, file_name)
