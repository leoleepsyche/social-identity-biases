import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

# Load model and tokenizer
tokenizer = BertTokenizer.from_pretrained('.\\Erlangshen-Roberta-110M-Sentiment')
model = BertForSequenceClassification.from_pretrained('.\\Erlangshen-Roberta-110M-Sentiment')
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load CSV file
df = pd.read_csv('.\\data\\all_data_berttopic_stm_control.csv')  # Replace with your filename
sentences = df['sentence'].astype(str).tolist()

# Batch prediction function
def predict_batch(text_batch):
    # Encode into model input format
    inputs = tokenizer(text_batch, return_tensors='pt', truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        preds = torch.argmax(probs, dim=1)
    return preds.tolist(), probs.tolist()

# Set batch size
batch_size = 32

# Store prediction results
all_labels = []
# tqdm for progress bar
for i in tqdm(range(0, len(sentences), batch_size), desc="Predicting"):
    batch_sentences = sentences[i:i+batch_size]
    labels, probs = predict_batch(batch_sentences)
    all_labels.extend(labels)

# Add results to DataFrame
df['sentiment_erlangshen'] = all_labels

# Save results
df.to_csv('.\\data\\all_data_berttopic_stm_control_sentiment.csv', index=False, encoding='utf-8-sig')
