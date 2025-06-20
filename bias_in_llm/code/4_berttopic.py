import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer


embedding_model = SentenceTransformer("text2vec-base-chinese")
topic_model = BERTopic(embedding_model=embedding_model, language="chinese")

df = pd.read_csv('./data/all_data.csv')
documents = df["sentence"].astype(str).tolist()
    
topics, probs = topic_model.fit_transform(documents)
df["berttopic"] = topics

df.to_csv('./data/all_data_berttopic.csv', index=False, encoding='utf-8-sig')

# print information of topics
print(topic_model.get_topic_info())
