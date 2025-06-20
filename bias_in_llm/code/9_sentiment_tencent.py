# -*- coding: utf-8 -*-

import json
import types
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
from tencentcloud.nlp.v20190408 import nlp_client, models
import pandas as pd
from tqdm import tqdm

# Instantiate a credential object; the parameters require Tencent Cloud SecretId and SecretKey.
# Be sure to keep your credentials secure—leaking them could compromise your entire account.
# The following code is for reference only. It is recommended to use a more secure approach to handle credentials.
# See: https://cloud.tencent.com/document/product/1278/85305
# You can obtain your credentials from: https://console.cloud.tencent.com/cam/capi
cred = credential.Credential("", "")
# Example using temporary credentials:
# cred = credential.Credential("SecretId", "SecretKey", "Token")

# Instantiate an optional HTTP profile; can be skipped if not needed
httpProfile = HttpProfile()
httpProfile.endpoint = "nlp.tencentcloudapi.com"

# Instantiate an optional client profile; can be skipped if not needed
clientProfile = ClientProfile()
clientProfile.httpProfile = httpProfile

# Instantiate the client object for the requested product. The clientProfile is optional.
client = nlp_client.NlpClient(cred, "", clientProfile)

def get_sentiment(text: str):
    try:
        # Instantiate a request object; each API corresponds to a request object
        req = models.AnalyzeSentimentRequest()
        params = {
            "Text": text
        }
        req.from_json_string(json.dumps(params))

        # The response returned is an instance of AnalyzeSentimentResponse, corresponding to the request object
        resp = client.AnalyzeSentiment(req)
        # Parse the JSON response
        result = json.loads(resp.to_json_string())
        sentiment = result["Sentiment"]
        return sentiment  # Output: positive
    except TencentCloudSDKException as err:
        return err

tqdm.pandas()

# Read CSV file
df = pd.read_csv('.\\data\\all_data_berttopic_stm_control_sentiment.csv')

# Perform sentiment analysis on each row
df['sentiment_tencent'] = df['sentence'].progress_apply(get_sentiment)

# Save the result to a new CSV file
df.to_csv(".\\data\\all_data_berttopic_stm_control_sentiment.csv", index=False, encoding='utf-8-sig')

print("Sentiment analysis completed")
