import pandas as pd
from cemotion import Cemotion
from tqdm import tqdm

# Load CSV file
df = pd.read_csv('.\\data\\all_data_berttopic_stm_control_sentiment.csv')  # Replace with your file name

# Initialize the cemotion model
cemo = Cemotion()

# Perform sentiment prediction for each row
tqdm.pandas()
df["cemotion_score"] = df["sentence"].progress_apply(lambda x: cemo.predict(str(x)))
df["sentiment_cemotion"] = df["cemotion_score"].apply(lambda x: "positive" if x >= 0.5 else "negative")

# Save the results
df.to_csv('.\\data\\all_data_berttopic_stm_control_sentiment.csv', index=False, encoding='utf-8-sig')
