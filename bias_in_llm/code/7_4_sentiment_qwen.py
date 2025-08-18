import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'

# 模型名称和加载
model_name = "/home/miniconda/hgmodel/Qwen3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.eval()

# 读取数据
df = pd.read_csv("./data/all_data_berttopic_stm_control_sentiment.csv")  # 替换为你的文件路径
sentences = df["sentence"].tolist()

# 情感分类函数（批量处理）
def batch_classify(sentences, batch_size=8):
    results = []
    for i in tqdm(range(0, len(sentences), batch_size)):
        batch = sentences[i:i + batch_size]
        prompts = [f"""请判断下面这句话的情感倾向。回答只能是一个单词，从["positive", "negative", "neutral"]中选取，不允许输出其他内容。
句子：{s}
情感：""" for s in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=10,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
            )

        for j in range(len(batch)):
            output = tokenizer.decode(outputs[j][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip().lower()
            if "positive" in output:
                results.append("positive")
            elif "negative" in output:
                results.append("negative")
            elif "neutral" in output:
                results.append("neutral")
            else:
                print(output)
                results.append("unknown")
    return results

# 执行分类
df["sentiment_qwen3"] = batch_classify(sentences, batch_size=200)  # 可根据显存调节 batch_size

# 保存结果
df.to_csv("./data/all_data_berttopic_stm_control_sentiment.csv", index=False, encoding='utf-8-sig')
print("分类完成，已保存为 classified_sentences.csv")
