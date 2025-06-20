import json
import os

# Here we use word segmentation as an example; for other algorithm API names and parameters, please refer to the documentation
from aliyunsdkalinlp.request.v20200629 import GetWsCustomizedChGeneralRequest

from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException
from aliyunsdkcore.acs_exception.exceptions import ServerException

import pandas as pd
from tqdm import tqdm

access_key_id = ''
access_key_secret = ''

# Create an AcsClient instance
client = AcsClient(
    access_key_id,
    access_key_secret,
    "cn-hangzhou"
)

def get_sentiment(text: str):
    """
    Call the Alibaba Cloud NLP API to perform sentiment analysis on Chinese text.

    Args:
        text (str): The Chinese text to be analyzed

    Returns:
        str: Sentiment classification result returned by the API. If failed, returns an error message.
    """
    request = GetWsCustomizedChGeneralRequest.GetWsCustomizedChGeneralRequest()
    request.set_action_name('GetSaChGeneral')  # Note: Your API may not be for segmentation, make sure to confirm this
    request.set_Text(text)
    request.set_OutType("1")
    request.set_ServiceCode("alinlp")
    request.set_TokenizerId("GENERAL_CHN")

    try:
        response = client.do_action_with_exception(request)
        result = json.loads(response)
        data_str = result.get("Data", "{}")
        data = json.loads(data_str)
        return data.get("result", {}).get("sentiment", "Unknown")
    except (ClientException, ServerException) as e:
        return f"Error: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON Error: {str(e)}"

tqdm.pandas()

# Read CSV
df = pd.read_csv('.\\data\\all_data_berttopic_stm_control.csv')

# Perform sentiment analysis for each row
df['sentiment_aliyun'] = df['sentence'].progress_apply(get_sentiment)

# Save results to a new CSV
df.to_csv(".\\data\\all_data_berttopic_stm_control_sentiment.csv", index=False, encoding='utf-8-sig')

print("Sentiment analysis completed")
