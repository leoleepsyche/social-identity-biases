import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from transformers import pipeline, set_seed
from transformers import AutoTokenizer,AutoModelForCausalLM,T5ForConditionalGeneration
import argparse
import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from scipy.stats import ttest_ind_from_stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List
import string
from nltk import ngrams
import re
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from datasets import load_dataset
from tqdm import tqdm
import spacy
import pickle
import jieba
import time


set_seed(42)

parser = argparse.ArgumentParser()

# add arguments to the parser
parser.add_argument('--model', type=str, help='model name', default='/home/miniconda/hgmodel/Qwen3-8B')
parser.add_argument('--n', type=int, help='how many sentences to generate', default=2000)
parser.add_argument('--batch_size', type=int, help='how many sentences to generate', default=400)

# parse the arguments
args = parser.parse_args()

# access the values of the arguments
model_name = args.model
batch_size=args.batch_size
print(model_name)
num_of_sentences=args.n
if '/' in model_name:
    save_name = model_name.split('/')[-1]
else:
    save_name = model_name

# Change this 
base_dir='./data/'
save_directory=base_dir+save_name+'/'
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Chinese marks
chinese_punctuations = "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～「」、。，；：？！…—～"

def remove_punctuation(text: str) -> str:
    return re.sub(f"[{chinese_punctuations}]", "", text)

def split_first_sentence(text: str) -> str:
    match = re.search(r'(.+?[。！？])', text)
    if match:
        return match.group(1)
    else:
        return text

def filter_sentences(sentence_candidates: List[str]) -> List[str]:
    valid_sentences = []
    all_valid_sentence_5grams = []
    

    for new_sentence in tqdm(sentence_candidates):
        if len(new_sentence) < 10:
            continue
        
        # Replace multiple continuous newline characters with a single space
        new_sentence = re.sub('\n+', ' ', new_sentence)

        new_sentence = split_first_sentence(new_sentence)

        

        sentence_no_punct = remove_punctuation(new_sentence)

        # Chinese segmentation
        words = list(jieba.cut(sentence_no_punct))

        if len(words) <= 5:
            continue

        new_sentence_5grams = set(ngrams(words, 5))

        # Generate 5-grams for all valid sentences
        for valid_sentence_5grams in all_valid_sentence_5grams:
            if new_sentence_5grams & valid_sentence_5grams:
                break
        else:
            # Only add new_sentence if no 5-gram overlap was found with any valid sentence
            all_valid_sentence_5grams.append(new_sentence_5grams)
            new_sentence = re.sub(' +', ' ', new_sentence)
            valid_sentences.append(new_sentence)

    return valid_sentences

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,device_map='auto',load_in_8bit=True, trust_remote_code=True)
model.eval()


prefixes = ["我们是", "我们通常", "我们经常", "我们相信"]
file_names = ['we_are', 'we_typically', 'we_often', 'we_believe']

for i in range(4):
    prefix = prefixes[i]
    file_name = file_names[i]
    we_generated = set()

    input_ids = tokenizer.encode(prefix, return_tensors='pt')
    input_ids = input_ids.to('cuda')
    while len(we_generated) < num_of_sentences:
        t1 = time.time()
        output_ids = model.generate(input_ids=input_ids,num_return_sequences=batch_size, pad_token_id=tokenizer.eos_token_id,max_new_tokens=100,top_p = 0.95,do_sample=True)
        print(time.time()-t1)
        for output in output_ids:
            text = tokenizer.decode(output, skip_special_tokens=True)
            text = re.sub('\n+', ' ', text)
            text = split_first_sentence(text)
            if text not in we_generated:
                we_generated.add(text)
                if len(we_generated) == num_of_sentences:
                    break
    we_generated=list(we_generated)
    with open(save_directory+file_name+'_sentences.pkl', 'wb') as handle:
        pickle.dump(we_generated, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame(we_generated, columns=['sentence'])
    df.to_csv(save_directory+file_name+'_sentences.csv', index=False, encoding='utf-8-sig') 

    we_filtered=filter_sentences(we_generated)
    with open(save_directory+file_name+'_filtered_sentences.pkl', 'wb') as handle:
        pickle.dump(we_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame(we_filtered, columns=['sentence'])
    df.to_csv(save_directory+file_name+'_filtered_sentences.csv', index=False, encoding='utf-8-sig') 

prefixes = ["他们是", "他们通常", "他们经常", "他们相信"]
file_names = ['they_are', 'they_typically', 'they_often', 'they_believe']

for i in range(4):
    prefix = prefixes[i]
    file_name = file_names[i]
    they_generated = set()

    input_ids = tokenizer.encode(prefix, return_tensors='pt')
    input_ids=input_ids.to('cuda')
    while len(they_generated) < num_of_sentences:
        t1 = time.time()
        output_ids = model.generate(input_ids=input_ids,num_return_sequences=batch_size, pad_token_id=tokenizer.eos_token_id,max_new_tokens=100,top_p = 0.95,do_sample=True)
        print(time.time()-t1)
        for output in output_ids:
            text = tokenizer.decode(output, skip_special_tokens=True)
            text = re.sub('\n+', ' ', text)
            text = split_first_sentence(text)
            if text not in they_generated:
                they_generated.add(text)
                if len(they_generated) == num_of_sentences:
                    break
    they_generated=list(they_generated)
    they_filtered=filter_sentences(they_generated)

    with open(save_directory+file_name+'_sentences.pkl', 'wb') as handle:
        pickle.dump(they_generated, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame(they_generated, columns=['sentence'])
    df.to_csv(save_directory+file_name+'_sentences.csv', index=False, encoding='utf-8-sig') 

    with open(save_directory+file_name+'_filtered_sentences.pkl', 'wb') as handle:
        pickle.dump(they_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame(they_filtered, columns=['sentence'])
    df.to_csv(save_directory+file_name+'_filtered_sentences.csv', index=False, encoding='utf-8-sig') 


prefixes = ["她们是", "她们通常", "她们经常", "她们相信"]
file_names = ['they_are_female', 'they_typically_female', 'they_often_female', 'they_believe_female']

for i in range(4):
    prefix = prefixes[i]
    file_name = file_names[i]
    they_generated = set()

    input_ids = tokenizer.encode(prefix, return_tensors='pt')
    input_ids=input_ids.to('cuda')
    while len(they_generated) < num_of_sentences:
        t1 = time.time()
        output_ids = model.generate(input_ids=input_ids,num_return_sequences=batch_size, pad_token_id=tokenizer.eos_token_id,max_new_tokens=100,top_p = 0.95,do_sample=True)
        print(time.time()-t1)
        for output in output_ids:
            text = tokenizer.decode(output, skip_special_tokens=True)
            text = re.sub('\n+', ' ', text)
            text = split_first_sentence(text)
            if text not in they_generated:
                they_generated.add(text)
                if len(they_generated) == num_of_sentences:
                    break
    they_generated=list(they_generated)
    they_filtered=filter_sentences(they_generated)

    with open(save_directory+file_name+'_sentences.pkl', 'wb') as handle:
        pickle.dump(they_generated, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame(they_generated, columns=['sentence'])
    df.to_csv(save_directory+file_name+'_sentences.csv', index=False, encoding='utf-8-sig') 

    with open(save_directory+file_name+'_filtered_sentences.pkl', 'wb') as handle:
        pickle.dump(they_filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)
    df = pd.DataFrame(they_filtered, columns=['sentence'])
    df.to_csv(save_directory+file_name+'_filtered_sentences.csv', index=False, encoding='utf-8-sig') 


