import pandas as pd
import numpy as np
import json
import torch
import time
import re
import requests
import urllib3
import os
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, AutoTokenizer
from openai import OpenAI
from dotenv import load_dotenv
import hmac
import hashlib
import base64
from urllib.parse import quote
import datetime

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Try to import cemotion, if not available, it will be handled gracefully
from cemotion import Cemotion
from snownlp import SnowNLP

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.nlp.v20190408 import nlp_client, models


from aliyunsdkalinlp.request.v20200629 import GetSaChGeneralRequest
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException



class SentimentAnalysis:
    def __init__(self, data_path="../result/2.topic_data_stm_WildChat-1M"):
        """
        Initialize sentiment analysis class with multiple methods

        Supported methods:
        - 'erlangshen': HuggingFace BERT model (Chinese) - 2-class classification
        - 'baidu': Baidu AI API (3-class) - Max 2048 bytes (~1000 Chinese chars)
        - 'openai': OpenAI/ChatGPT API (configurable)
        - 'cemotion': Cemotion Chinese emotion analysis (2-class)
        - 'snownlp': SnowNLP Chinese sentiment analysis (3-class with neutral)
        - 'tencent': Tencent Cloud API (3-class) - Max 200 characters
        - 'aliyun': Aliyun Machine Learning API (3-class) - Max 1000 characters
        """
        self.data_path = data_path
        self.df = None
        self.methods = {}
        self.current_model_info = {}  # Store current model information
        load_dotenv()
        # Set HuggingFace cache directory from environment variable
        hf_cache_dir = os.getenv('HF_CACHE_DIR', 'D:\\HuggingFace')
        os.environ['HF_HOME'] = hf_cache_dir
        os.environ['TRANSFORMERS_CACHE'] = os.path.join(hf_cache_dir, 'Models')

        # Method configurations - using environment variables for sensitive data
        self.method_configs = {
            'erlangshen': {
                'model_name': "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment",
                'cache_dir': os.path.join(hf_cache_dir, 'Models')
            },
            'baidu': {
                'api_key': os.getenv('BAIDU_API_KEY', ''),
                'secret_key': os.getenv('BAIDU_SECRET_KEY', ''),
                'output_format': 'three_class'  # pessimistic, neutral, optimistic
            },
            'openai': {
                'api_key': os.getenv('OPENAI_API_KEY', ''),
                'base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
                'use_three_class': True  # Can be changed to False for 2-class
            },
            'cemotion': {
                'threshold': 0.5,  # 0-0.5: negative, 0.5-1: positive
                'use_binary': True  # Always binary classification for cemotion
            },
            'snownlp': {
                'threshold': 0.5,  # 0-0.5: negative, 0.5-1: positive
                'neutral_range': 0.1  # +/- range around threshold for neutral classification
            },
            'tencent': {
                'secret_id': os.getenv('TENCENT_SECRET_ID', ''),
                'secret_key': os.getenv('TENCENT_SECRET_KEY', ''),
                'endpoint': 'https://nlp.tencentcloudapi.com',
                'action': 'AnalyzeSentiment',
                'version': '2019-04-08',
                'output_format': 'three_class'  # positive, negative, neutral
            },
            'aliyun': {
                'access_key_id': os.getenv('ALIYUN_ACCESS_KEY_ID', ''),
                'access_key_secret': os.getenv('ALIYUN_ACCESS_KEY_SECRET', ''),
                'region': 'cn-hangzhou',
                'action': 'SaChGeneral',
                'service_code': 'alinlp',
                'output_format': 'three_class'  # 正面, 负面, 中性
            }
        }

    def load_data(self):
        """Load data with topic information"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
        
        # === test mode ===
        self.df = self.df.head(5)
        
        print(f"Data loaded successfully, {len(self.df)} records in total")

        # Check required columns
        required_cols = ['text']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        return self.df

    def init_method(self, method):
        """Initialize specific sentiment analysis method"""
        if method in self.methods:
            return self.methods[method]

        if method == 'erlangshen':
            print(f"Loading Erlangshen model: {self.method_configs['erlangshen']['model_name']}")
            print(f"Cache directory: {self.method_configs['erlangshen']['cache_dir']}")

            tokenizer = BertTokenizer.from_pretrained(
                self.method_configs['erlangshen']['model_name'],
                cache_dir=self.method_configs['erlangshen']['cache_dir']
            )
            model = BertForSequenceClassification.from_pretrained(
                self.method_configs['erlangshen']['model_name'],
                cache_dir=self.method_configs['erlangshen']['cache_dir']
            )

            print("Erlangshen model: 2-class classification (0=negative, 1=positive)")

            self.methods[method] = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': self.method_configs['erlangshen']['model_name']
            }

            # Store model info for file naming
            self.current_model_info[method] = {
                'model_name': self.method_configs['erlangshen']['model_name']
            }

            print("Erlangshen model loaded successfully")

        elif method == 'baidu':
            # Check if API keys are provided
            if not self.method_configs['baidu']['api_key'] or not self.method_configs['baidu']['secret_key']:
                raise ValueError(
                    "Baidu API keys not found. Please set BAIDU_API_KEY and BAIDU_SECRET_KEY environment variables.")

            print("Initializing Baidu API...")
            access_token = self._get_baidu_access_token()
            api_url = f"https://aip.baidubce.com/rpc/2.0/nlp/v1/sentiment_classify?charset=UTF-8&access_token={access_token}"
            self.methods[method] = {'api_url': api_url}

            # Store model info for file naming
            self.current_model_info[method] = {
                'model_name': 'baidu-sentiment-api',
                'api_version': 'v1'
            }

            print("Baidu API initialized successfully")

        elif method == 'openai':
            # Check if API key is provided
            if not self.method_configs['openai']['api_key']:
                raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

            print("Initializing OpenAI API...")
            client = OpenAI(
                api_key=self.method_configs['openai']['api_key'],
                base_url=self.method_configs['openai']['base_url']
            )

            # Configure prompt based on class setting
            if self.method_configs['openai']['use_three_class']:
                system_prompt = """You are a professional Chinese sentiment analysis assistant. Please analyze the sentiment of given text and respond in the following format:        
                                    Format: sentiment: [number] | confidence: [decimal 0-1]

                                    Where sentiment values are:
                                    - 0: Negative sentiment (pessimistic, critical, dissatisfied, angry, etc.)
                                    - 1: Neutral sentiment (objective, factual, neither positive nor negative)
                                    - 2: Positive sentiment (optimistic, praise, satisfied, happy, etc.)

                                    Confidence is your certainty level, range 0-1.

                                    Examples:
                                    Input: This product quality is terrible
                                    Output: sentiment: 0 | confidence: 0.9

                                    Input: Today the weather is nice
                                    Output: sentiment: 2 | confidence: 0.8

                                    Input: This is a factual report
                                    Output: sentiment: 1 | confidence: 0.7

                                    Please only return the specified format, no additional explanation."""
            else:
                system_prompt = """You are a professional Chinese sentiment analysis assistant. Please analyze the sentiment of given text and respond in the following format:        
                                    Format: sentiment: [number] | confidence: [decimal 0-1]

                                    Where sentiment values are:
                                    - 0: Negative sentiment (pessimistic, critical, dissatisfied, angry, etc.)
                                    - 1: Positive sentiment (optimistic, praise, satisfied, happy, etc.)

                                    Confidence is your certainty level, range 0-1.

                                    Examples:
                                    Input: This product quality is terrible
                                    Output: sentiment: 0 | confidence: 0.9

                                    Input: Today the weather is nice
                                    Output: sentiment: 1 | confidence: 0.8

                                    Please only return the specified format, no additional explanation."""

            self.methods[method] = {
                'client': client,
                'system_prompt': system_prompt,
                'use_three_class': self.method_configs['openai']['use_three_class'],
                'model_name': self.method_configs['openai']['model']
            }

            # Store model info for file naming
            self.current_model_info[method] = {
                'model_name': self.method_configs['openai']['model'],
                'use_three_class': self.method_configs['openai']['use_three_class'],
                'base_url': self.method_configs['openai']['base_url']
            }

            print("OpenAI API initialized successfully")

        elif method == 'cemotion':
            print("Initializing Cemotion method...")
            try:
                cemotion_model = Cemotion()
                self.methods[method] = {
                    'model': cemotion_model,
                    'threshold': self.method_configs['cemotion']['threshold']
                }

                # Store model info for file naming
                self.current_model_info[method] = {
                    'model_name': 'cemotion-chinese-2class',
                    'threshold': self.method_configs['cemotion']['threshold'],
                    'version': 'binary'
                }

                print("Cemotion method initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Cemotion: {e}")
                raise

        elif method == 'snownlp':
            print("Initializing SnowNLP method...")
            self.methods[method] = {
                'threshold': self.method_configs['snownlp']['threshold'],
                'neutral_range': self.method_configs['snownlp']['neutral_range']
            }

            # Store model info for file naming
            self.current_model_info[method] = {
                'model_name': 'snownlp-chinese',
                'threshold': self.method_configs['snownlp']['threshold'],
                'neutral_range': self.method_configs['snownlp']['neutral_range']
            }

            print("SnowNLP method initialized successfully")

        elif method == 'tencent':
            # Check if API keys are provided
            if not self.method_configs['tencent']['secret_id'] or not self.method_configs['tencent']['secret_key']:
                raise ValueError(
                    "Tencent API keys not found. Please set TENCENT_SECRET_ID and TENCENT_SECRET_KEY environment variables.")

            print("Initializing Tencent Cloud method...")
            self.methods[method] = {
                'secret_id': self.method_configs['tencent']['secret_id'],
                'secret_key': self.method_configs['tencent']['secret_key'],
                'endpoint': self.method_configs['tencent']['endpoint'],
                'action': self.method_configs['tencent']['action'],
                'version': self.method_configs['tencent']['version'],
                'output_format': self.method_configs['tencent']['output_format']
            }

            # Store model info for file naming
            self.current_model_info[method] = {
                'model_name': 'tencent-sentiment-api',
                'action': self.method_configs['tencent']['action']
            }

            print("Tencent Cloud method initialized successfully")

        elif method == 'aliyun':
            # Check if SDK is available
            
            # Check if API keys are provided
            if not self.method_configs['aliyun']['access_key_id'] or not self.method_configs['aliyun']['access_key_secret']:
                raise ValueError(
                    "Aliyun API keys not found. Please set ALIYUN_ACCESS_KEY_ID and ALIYUN_ACCESS_KEY_SECRET environment variables.")

            print("Initializing Aliyun API...")
            self.methods[method] = {
                'access_key_id': self.method_configs['aliyun']['access_key_id'],
                'access_key_secret': self.method_configs['aliyun']['access_key_secret'],
                'region': self.method_configs['aliyun']['region'],
                'action': self.method_configs['aliyun']['action'],
                'service_code': self.method_configs['aliyun']['service_code']
            }

            # Store model info for file naming
            self.current_model_info[method] = {
                'model_name': 'aliyun-sentiment-api',
                'action': self.method_configs['aliyun']['action']
            }

            print("Aliyun API initialized successfully")

        return self.methods[method]

    def _get_model_identifier(self, method):
        """Generate model identifier for file naming"""
        if method not in self.current_model_info:
            return method

        info = self.current_model_info[method]

        if method == 'erlangshen':
            # Extract simplified model name
            model_name = info['model_name'].split('/')[-1]  # Get last part after /
            return f"bert_{model_name}_2class"

        elif method == 'openai':
            model_name = info['model_name'].replace('-', '_')
            class_suffix = "3class" if info['use_three_class'] else "2class"
            return f"openai_{model_name}_{class_suffix}"

        elif method == 'baidu':
            return f"baidu_{info['model_name']}"

        elif method == 'cemotion':
            threshold_str = str(info['threshold']).replace('.', '_')
            return f"cemotion_{info['model_name']}_threshold_{threshold_str}"

        elif method == 'snownlp':
            threshold_str = str(info['threshold']).replace('.', '_')
            neutral_str = str(info['neutral_range']).replace('.', '_')
            return f"snownlp_{info['model_name']}_threshold_{threshold_str}_neutral_{neutral_str}"

        elif method == 'tencent':
            return f"tencent_{info['model_name']}"

        elif method == 'aliyun':
            return f"aliyun_{info['model_name']}"

        else:
            return method

    def predict_sentiment_erlangshen(self, text):
        """Predict sentiment using Erlangshen model (2-class: 0=negative, 1=positive)"""
        method_obj = self.init_method('erlangshen')
        model = method_obj['model']
        tokenizer = method_obj['tokenizer']

        inputs = tokenizer(text, add_special_tokens=True, max_length=512, padding=True, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            output = model(**inputs)
            predictions = torch.nn.functional.softmax(output.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Erlangshen model: 0=negative, 1=positive (binary classification)
        return '正面' if predicted_class == 1 else '负面'

    def predict_sentiment_baidu(self, text):
        """Predict sentiment using Baidu API (3-class, max 2048 bytes)"""
        method_obj = self.init_method('baidu')
        api_url = method_obj['api_url']

        # Baidu API limit: 2048 bytes (approximately 1000 Chinese characters)
        if len(text.encode('utf-8')) > 2048:
            # Truncate by characters to be safe
            while len(text.encode('utf-8')) > 2048 and len(text) > 0:
                text = text[:-1]
            print(f"Warning: Text truncated to fit Baidu API limit (2048 bytes)")

        payload = json.dumps({"text": text}, ensure_ascii=False)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

        response = requests.post(api_url, headers=headers, data=payload.encode("UTF-8"), verify=False, timeout=30)
        result = response.json()

        # Check for API errors
        if 'error_code' in result:
            error_msg = result.get('error_msg', 'Unknown error')
            raise Exception(f"Baidu API Error {result['error_code']}: {error_msg}")

        if 'items' in result and len(result['items']) > 0:
            sentiment = result['items'][0].get('sentiment', 1)  # 0:negative, 1:neutral, 2:positive
            if sentiment == 0:
                return '负面'
            elif sentiment == 2:
                return '正面'
            else:
                return '中性'
        else:
            raise Exception("Baidu API: No valid response data")

    def predict_sentiment_openai(self, text):
        """Predict sentiment using OpenAI API (configurable 2-class or 3-class)"""
        method_obj = self.init_method('openai')
        client = method_obj['client']
        system_prompt = method_obj['system_prompt']
        use_three_class = method_obj['use_three_class']

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please analyze the sentiment of the following text:\n{text}"}
        ]

        try:
            response = client.chat.completions.create(
                model=self.method_configs['openai']['model'],
                messages=messages
            )

            result_text = response.choices[0].message.content.strip()
            return self._parse_openai_response(result_text, use_three_class)
        except Exception as e:
            raise Exception(f"OpenAI API Error: {str(e)}")

    def _parse_openai_response(self, result_text, use_three_class):
        """Parse OpenAI response with dynamic class handling"""
        sentiment_match = re.search(r'sentiment:\s*(\d+)', result_text)

        if sentiment_match:
            sentiment = int(sentiment_match.group(1))

            if use_three_class:
                # 3-class: 0=negative, 1=neutral, 2=positive
                if sentiment == 0:
                    return '负面'
                elif sentiment == 2:
                    return '正面'
                else:
                    return '中性'
            else:
                # 2-class: 0=negative, 1=positive
                return '正面' if sentiment == 1 else '负面'
        else:
            return self._parse_openai_fallback(result_text, use_three_class)

    def _parse_openai_fallback(self, result_text, use_three_class):
        """Fallback parsing for OpenAI response"""
        result_lower = result_text.lower()

        if any(word in result_lower for word in ['negative', 'pessimistic', '负面', '消极', '0']):
            return '负面'
        elif any(word in result_lower for word in ['positive', 'optimistic', '正面', '积极']) or '2' in result_lower:
            return '正面'
        elif use_three_class and any(word in result_lower for word in ['neutral', 'objective', '中性', '客观', '1']):
            return '中性'
        else:
            return '负面'  # default negative

    def predict_sentiment_cemotion(self, text):
        method_obj = self.init_method('cemotion')
        cemotion_model = method_obj['model']
        threshold = method_obj['threshold']

        score = cemotion_model.predict(text)

        if score > threshold:
            return '正面'
        else:
            return '负面'

    def predict_sentiment_snownlp(self, text):
        """Predict sentiment using SnowNLP (0-1 score, supports neutral classification)"""
        method_obj = self.init_method('snownlp')
        threshold = method_obj['threshold']
        neutral_range = method_obj['neutral_range']

        # SnowNLP sentiment analysis
        s = SnowNLP(text)
        score = s.sentiments  # Returns a float between 0 and 1

        # Classification with neutral zone
        if score > (threshold + neutral_range):
            return '正面'
        elif score < (threshold - neutral_range):
            return '负面'
        else:
            return '中性'

    def predict_sentiment_tencent(self, text):
        """Predict sentiment using Tencent Cloud API (3-class, max 200 characters)"""
        method_obj = self.init_method('tencent')
        secret_id = method_obj['secret_id']
        secret_key = method_obj['secret_key']

        # Tencent Cloud API limit: 200 characters
        if len(text) > 200:
            text = text[:200]
        original_proxies = {}
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ[var]
                del os.environ[var]
        cred = credential.Credential(secret_id, secret_key)
        httpProfile = HttpProfile()
        httpProfile.endpoint = "nlp.tencentcloudapi.com"
        httpProfile.proxy = None
        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        clientProfile.httpProfile.reqTimeout = 30


        client = nlp_client.NlpClient(cred, "", clientProfile)

        req = models.AnalyzeSentimentRequest()
        params = {
            "Text": text
        }
        req.from_json_string(json.dumps(params))

        resp = client.AnalyzeSentiment(req)

        response_dict = json.loads(resp.to_json_string())
        sentiment = response_dict.get('Sentiment', 'neutral')
        os.environ.update(original_proxies)
        # Map English to Chinese
        if sentiment == 'positive':
            return '正面'
        elif sentiment == 'negative':
            return '负面'
        else:
            return '中性'

    def predict_sentiment_aliyun(self, text):
        """Predict sentiment using Aliyun API (3-class, max 1000 characters)"""
        method_obj = self.init_method('aliyun')
        access_key_id = method_obj['access_key_id']
        access_key_secret = method_obj['access_key_secret']
        region = method_obj['region']
        service_code = method_obj['service_code']

        # Aliyun API limit: 1000 characters
        if len(text) > 1000:
            text = text[:1000]
        try:
            client = AcsClient(access_key_id, access_key_secret, region)

            request = GetSaChGeneralRequest.GetSaChGeneralRequest()
            request.set_Text(text)
            request.set_ServiceCode(service_code)

            response = client.do_action_with_exception(request)
            resp_obj = json.loads(response)

            if 'Data' in resp_obj:
                sentiment_result = resp_obj['Data']
                if 'result' in sentiment_result:
                    sentiment = sentiment_result['result']['sentiment']
                    return sentiment
            else:
                raise Exception("Aliyun API: No Data field in response")
        except (ClientException, ServerException) as e:
            raise Exception(f"Aliyun API Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Aliyun API Error: {str(e)}")


    def analyze_sentiment_batch(self, method='erlangshen'):
        """
        Batch sentiment analysis with dynamic class detection

        Args:
            method: 'erlangshen', 'baidu', 'openai', 'cemotion', 'snownlp', 'tencent', or 'aliyun'
        """
        if self.df is None:
            self.load_data()

        print(f"Performing sentiment analysis using {method.upper()} method...")

        # Select prediction function
        predict_funcs = {
            'erlangshen': self.predict_sentiment_erlangshen,
            'baidu': self.predict_sentiment_baidu,
            'openai': self.predict_sentiment_openai,
            'cemotion': self.predict_sentiment_cemotion,
            'snownlp': self.predict_sentiment_snownlp,
            'tencent': self.predict_sentiment_tencent,
            'aliyun': self.predict_sentiment_aliyun
        }

        predict_func = predict_funcs[method]

        results = []
        texts = self.df['text'].tolist()

        # Add progress bar
        for i in tqdm(range(len(texts)), desc=f"{method.upper()} sentiment analysis"):
            text = str(texts[i])
            result = predict_func(text)
            results.append(result)

            # Add delay for API methods
            if method in ['baidu', 'openai', 'tencent', 'aliyun']:
                time.sleep(0.5)

        # Save sentiment results
        self.df['sentiment'] = results

        print(f"{method.upper()} sentiment analysis completed")
        return results

    def save_results(self, method, output_dir="../result/"):
        """Save results with model-specific filename"""
        if self.df is None:
            raise ValueError("No data to save")

        # Generate model-specific identifier
        model_identifier = self._get_model_identifier(method)
        output_path = f"{output_dir}3.sentiment_data_{model_identifier}.csv"

        self.df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Results saved to: {output_path}")
        return output_path

    def analyze_sentiment_statistics(self, method='erlangshen'):
        """Analyze sentiment statistics for specific method"""
        if 'sentiment' not in self.df.columns:
            print(f"Warning: {method} results not found. Run analysis first.")
            return

        # Calculate statistics
        positive_count = (self.df['sentiment'] == '正面').sum()
        negative_count = (self.df['sentiment'] == '负面').sum()
        neutral_count = (self.df['sentiment'] == '中性').sum()
        total_count = len(self.df)

        print(f"\n=== {method.upper()} Statistics ===")
        print(f"Positive count: {positive_count} ({positive_count/total_count:.4f})")
        print(f"Negative count: {negative_count} ({negative_count/total_count:.4f})")
        print(f"Neutral count: {neutral_count} ({neutral_count/total_count:.4f})")

        # Group statistics if available
        if 'ingroup' in self.df.columns:
            ingroup_stats = self.df[self.df['ingroup'] == 1]
            outgroup_stats = self.df[self.df['ingroup'] == 0]

            ingroup_pos_rate = (ingroup_stats['sentiment'] == '正面').mean()
            outgroup_pos_rate = (outgroup_stats['sentiment'] == '正面').mean()

            print(f"Ingroup positive rate: {ingroup_pos_rate:.4f}")
            print(f"Outgroup positive rate: {outgroup_pos_rate:.4f}")

    def _get_baidu_access_token(self):
        """Get Baidu API access token"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.method_configs['baidu']['api_key'],
            "client_secret": self.method_configs['baidu']['secret_key']
        }
        return str(requests.post(url, params=params).json().get("access_token"))


def main():
    """Main function for testing"""
    methods_to_run = ['aliyun']  # Change this to ['erlangshen', 'baidu', 'openai', 'cemotion', 'snownlp', 'tencent', 'aliyun'] to run all

    for method in methods_to_run:
        print(f"\n{'=' * 50}")
        print(f"Running {method.upper()} sentiment analysis")
        print(f"{'=' * 50}")

        sa = SentimentAnalysis(data_path="../result/2.topic_data_stm_WildChat-1M.csv")
        sa.load_data()
        sa.analyze_sentiment_batch(method=method)
        sa.analyze_sentiment_statistics(method=method)
        sa.save_results(method=method)

    print("\nAll sentiment analysis completed!")


if __name__ == "__main__":
    main()