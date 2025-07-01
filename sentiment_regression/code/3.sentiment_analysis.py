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
import logging
from functools import wraps

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Setup logging for retry mechanism
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


def retry_api_call(max_retries=3, delay=1.0, backoff=2.0):
    """
    Retry decorator for API calls with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            current_delay = delay
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"API call {func.__name__} failed after {max_retries} retries: {str(e)}")
                        # Return default values instead of raising exception
                        return_score = kwargs.get('return_score', False)
                        if return_score:
                            return '负面', {'error': True, 'message': str(e)}
                        else:
                            return '负面'  # Default to negative sentiment
                    
                    logger.warning(f"API call {func.__name__} failed (attempt {retries}/{max_retries}): {str(e)}")
                    logger.info(f"Retrying in {current_delay} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # Should never reach here, but just in case
            return_score = kwargs.get('return_score', False)
            if return_score:
                return '负面', {'error': True, 'message': 'Max retries exceeded'}
            else:
                return '负面'
                
        return wrapper
    return decorator


class SentimentAnalysis:
    def __init__(self, data_path="../result/2.topic_data_stm_WildChat-1M"):
        """Initialize sentiment analysis with multiple API methods"""
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
                'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
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
        
        # Try to load from cumulative results file first
        cumulative_path = "../result/3.sentiment_data_all_methods.csv"
        if os.path.exists(cumulative_path):
            print(f"Found existing cumulative results file: {cumulative_path}")
            print("Loading from cumulative results to preserve previous analysis...")
            self.df = pd.read_csv(cumulative_path)
        else:
            # Load from original data file
            self.df = pd.read_csv(self.data_path)
        
        # === test mode ===
        # self.df = self.df.head(5)
        
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

            # Configure prompt for 3-class sentiment analysis
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

            self.methods[method] = {
                'client': client,
                'system_prompt': system_prompt,
                'model_name': self.method_configs['openai']['model']
            }

            # Store model info for file naming
            self.current_model_info[method] = {
                'model_name': self.method_configs['openai']['model'],
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
        info = self.current_model_info.get(method, {})
        
        identifiers = {
            'erlangshen': lambda: f"erlangshen_{info['model_name'].split('/')[-1]}_2class",
            'openai': lambda: f"openai_{info['model_name'].replace('-', '_')}_3class",
            'baidu': lambda: f"baidu_{info['model_name']}",
            'cemotion': lambda: f"cemotion_{info['model_name']}_threshold_{str(info['threshold']).replace('.', '_')}",
            'snownlp': lambda: f"snownlp_{info['model_name']}_threshold_{str(info['threshold']).replace('.', '_')}_neutral_{str(info['neutral_range']).replace('.', '_')}",
            'tencent': lambda: f"tencent_{info['model_name']}",
            'aliyun': lambda: f"aliyun_{info['model_name']}"
        }
        
        return identifiers.get(method, lambda: method)() if info else method

    def predict_sentiment_erlangshen(self, text, return_score=False):
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
        sentiment_label = '正面' if predicted_class == 1 else '负面'
        
        if return_score:
            # Return probabilities for both classes
            prob_negative = predictions[0][0].item()
            prob_positive = predictions[0][1].item()
            return sentiment_label, {'negative_prob': prob_negative, 'positive_prob': prob_positive}
        else:
            return sentiment_label

    @retry_api_call(max_retries=3, delay=1.0)
    def predict_sentiment_baidu(self, text, return_score=False):
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
            item = result['items'][0]
            sentiment = item.get('sentiment', 1)  # 0:negative, 1:neutral, 2:positive
            
            # Extract probability scores
            positive_prob = item.get('positive_prob', 0.0)
            negative_prob = item.get('negative_prob', 0.0)
            
            if sentiment == 0:
                sentiment_label = '负面'
            elif sentiment == 2:
                sentiment_label = '正面'
            else:
                sentiment_label = '中性'
                
            if return_score:
                return sentiment_label, {'positive_prob': positive_prob, 'negative_prob': negative_prob}
            else:
                return sentiment_label
        else:
            raise Exception("Baidu API: No valid response data")

    @retry_api_call(max_retries=3, delay=1.0)
    def predict_sentiment_openai(self, text, return_score=False):
        """Predict sentiment using OpenAI API (3-class classification)"""
        method_obj = self.init_method('openai')
        client = method_obj['client']
        system_prompt = method_obj['system_prompt']

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
            return self._parse_openai_response(result_text, return_score)
        except Exception as e:
            raise Exception(f"OpenAI API Error: {str(e)}")

    def _parse_openai_response(self, result_text, return_score=False):
        """Parse OpenAI response (3-class classification)"""
        sentiment_match = re.search(r'sentiment:\s*(\d+)', result_text)
        confidence_match = re.search(r'confidence:\s*([\d.]+)', result_text)
        
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0

        if sentiment_match:
            sentiment = int(sentiment_match.group(1))
            
            # 3-class: 0=negative, 1=neutral, 2=positive
            if sentiment == 0:
                sentiment_label = '负面'
            elif sentiment == 2:
                sentiment_label = '正面'
            else:
                sentiment_label = '中性'
                
            if return_score:
                return sentiment_label, {'confidence': confidence}
            else:
                return sentiment_label
        else:
            return self._parse_openai_fallback(result_text, return_score)

    def _parse_openai_fallback(self, result_text, return_score=False):
        """Fallback parsing for OpenAI response (3-class)"""
        result_lower = result_text.lower()

        if any(word in result_lower for word in ['negative', 'pessimistic', '负面', '消极', '0']):
            sentiment_label = '负面'
        elif any(word in result_lower for word in ['positive', 'optimistic', '正面', '积极']) or '2' in result_lower:
            sentiment_label = '正面'
        elif any(word in result_lower for word in ['neutral', 'objective', '中性', '客观', '1']):
            sentiment_label = '中性'
        else:
            sentiment_label = '负面'  # default negative
            
        if return_score:
            return sentiment_label, {'confidence': 0.0}  # Default confidence when parsing fails
        else:
            return sentiment_label

    def predict_sentiment_cemotion(self, text, return_score=False):
        """Predict sentiment using Cemotion model
        
        Args:
            text: Input text
            return_score: If True, return (sentiment, score), else return only sentiment
        """
        method_obj = self.init_method('cemotion')
        cemotion_model = method_obj['model']
        threshold = method_obj['threshold']

        score = cemotion_model.predict(text)

        if score > threshold:
            sentiment = '正面'
        else:
            sentiment = '负面'
            
        if return_score:
            return sentiment, score
        else:
            return sentiment

    def predict_sentiment_snownlp(self, text, return_score=False):
        """Predict sentiment using SnowNLP (0-1 score, supports neutral classification)"""
        method_obj = self.init_method('snownlp')
        threshold = method_obj['threshold']
        neutral_range = method_obj['neutral_range']

        # SnowNLP sentiment analysis
        s = SnowNLP(text)
        score = s.sentiments  # Returns a float between 0 and 1

        # Classification with neutral zone
        if score > (threshold + neutral_range):
            sentiment = '正面'
        elif score < (threshold - neutral_range):
            sentiment = '负面'
        else:
            sentiment = '中性'
            
        if return_score:
            return sentiment, score
        else:
            return sentiment

    @retry_api_call(max_retries=3, delay=1.0)
    def predict_sentiment_tencent(self, text, return_score=False):
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
        
        # Extract probability scores
        positive_score = response_dict.get('Positive', 0.0)
        neutral_score = response_dict.get('Neutral', 0.0)
        negative_score = response_dict.get('Negative', 0.0)
        
        os.environ.update(original_proxies)
        
        # Map English to Chinese
        if sentiment == 'positive':
            sentiment_label = '正面'
        elif sentiment == 'negative':
            sentiment_label = '负面'
        else:
            sentiment_label = '中性'
            
        if return_score:
            return sentiment_label, {
                'positive': positive_score,
                'neutral': neutral_score,
                'negative': negative_score
            }
        else:
            return sentiment_label

    @retry_api_call(max_retries=3, delay=1.0)
    def predict_sentiment_aliyun(self, text, return_score=False):
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
                sentiment_result = json.loads(resp_obj['Data'])
                if 'result' in sentiment_result:
                    result = sentiment_result['result']
                    sentiment = result['sentiment']
                    
                    # Extract probability scores
                    positive_prob = result.get('positive_prob', 0.0)
                    negative_prob = result.get('negative_prob', 0.0)
                    
                    if return_score:
                        return sentiment, {'positive_prob': positive_prob, 'negative_prob': negative_prob}
                    else:
                        return sentiment
            else:
                raise Exception("Aliyun API: No Data field in response")
        except (ClientException, ServerException) as e:
            raise Exception(f"Aliyun API Error: {str(e)}")
        except Exception as e:
            raise Exception(f"Aliyun API Error: {str(e)}")


    def _extract_score_safely(self, score, key, default=0.0):
        """Safely extract score value with error handling"""
        if isinstance(score, dict):
            return score.get(key, default) if not score.get('error', False) else default
        return score if key == 'raw' else default

    def _save_method_scores(self, method, scores):
        """Save method-specific scores with unified error handling"""
        score_configs = {
            'erlangshen': [
                ('erlangshen_negative_prob', 'negative_prob'),
                ('erlangshen_positive_prob', 'positive_prob')
            ],
            'cemotion': [('cemotion_score', 'raw')],
            'snownlp': [('snownlp_score', 'raw')],
            'baidu': [
                ('baidu_positive_prob', 'positive_prob'),
                ('baidu_negative_prob', 'negative_prob')
            ],
            'openai': [('openai_confidence', 'confidence')],
            'tencent': [
                ('tencent_positive', 'positive'),
                ('tencent_neutral', 'neutral'),
                ('tencent_negative', 'negative')
            ],
            'aliyun': [
                ('aliyun_positive_prob', 'positive_prob'),
                ('aliyun_negative_prob', 'negative_prob')
            ]
        }
        
        if method in score_configs:
            for col_name, score_key in score_configs[method]:
                values = [self._extract_score_safely(score, score_key) for score in scores]
                self.df[col_name] = values
            
            col_names = [col for col, _ in score_configs[method]]
            print(f"{method.upper()} scores saved to {', '.join(col_names)} columns")

    def analyze_sentiment_batch(self, method='erlangshen'):
        """Batch sentiment analysis with unified processing"""
        if self.df is None:
            self.load_data()

        print(f"Performing sentiment analysis using {method.upper()} method...")

        # Get prediction function
        predict_func = getattr(self, f'predict_sentiment_{method}')
        
        results, scores, error_count = [], [], 0
        api_methods = ['baidu', 'openai', 'tencent', 'aliyun']
        
        # Process each text
        for i, text in enumerate(tqdm(self.df['text'].tolist(), desc=f"{method.upper()} analysis")):
            sentiment, score = predict_func(str(text), return_score=True)
            
            # Check for errors
            if isinstance(score, dict) and score.get('error', False):
                error_count += 1
                logger.warning(f"Error in text {i+1}: {score.get('message', 'Unknown error')}")
            
            results.append(sentiment)
            scores.append(score)
            
            # Add delay for API methods
            if method in api_methods:
                time.sleep(0.5)

        # Save results
        self.df[f'sentiment_{method}'] = results
        print(f"Sentiment results saved to 'sentiment_{method}' column")
        
        # Report statistics
        if error_count > 0:
            print(f"Warning: {error_count}/{len(results)} API calls failed")
        else:
            print("All API calls completed successfully")
        
        # Save scores
        self._save_method_scores(method, scores)
        print(f"{method.upper()} sentiment analysis completed")
        return results

    def save_results(self, method, output_dir="../result/"):
        """Save results to cumulative sentiment analysis file"""
        if self.df is None:
            raise ValueError("No data to save")

        output_path = f"{output_dir}3.sentiment_data_all_methods.csv"
        
        # Get all new columns for this method
        new_columns = [col for col in self.df.columns if col.startswith(f'{method}_') or col == f'sentiment_{method}']
        
        try:
            if os.path.exists(output_path):
                existing_df = pd.read_csv(output_path, encoding='utf-8')
                print(f"Loading existing results from: {output_path}")
                
                if len(existing_df) == len(self.df):
                    # Update existing columns
                    for col in new_columns:
                        existing_df[col] = self.df[col]
                    existing_df.to_csv(output_path, index=False, encoding='utf-8')
                    print(f"Updated with {method} analysis")
                else:
                    print("Data length mismatch. Creating new file.")
                    self.df.to_csv(output_path, index=False, encoding='utf-8')
            else:
                # Create new file
                self.df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"Created new results file")
                
            print(f"Added columns: {new_columns}")
            
        except Exception as e:
            print(f"Error saving file: {e}")
            print("Creating backup results file.")
            self.df.to_csv(output_path, index=False, encoding='utf-8')
            
        return output_path

    def analyze_sentiment_statistics(self, method='erlangshen'):
        """Analyze sentiment statistics for specific method"""
        sentiment_col = f'sentiment_{method}'
        if sentiment_col not in self.df.columns:
            print(f"Warning: {method} results not found. Run analysis first.")
            return

        # Calculate basic statistics
        sentiments = self.df[sentiment_col].value_counts()
        total = len(self.df)
        
        print(f"\n=== {method.upper()} Statistics ===")
        for sentiment, count in sentiments.items():
            print(f"{sentiment}: {count} ({count/total:.3f})")
        
        # Group statistics if available
        if 'group' in self.df.columns:
            for group in self.df['group'].unique():
                group_data = self.df[self.df['group'] == group]
                pos_rate = (group_data[sentiment_col] == '正面').mean()
                print(f"{group} positive rate: {pos_rate:.3f}")

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
    """Run sentiment analysis for specified methods"""
    #'baidu','aliyun','openai','tencent'
    #['erlangshen', 'cemotion', 'snownlp']
    methods_to_run = ['erlangshen', 'cemotion', 'snownlp', 'baidu','aliyun','openai','tencent']
    
    print(f"=== Sentiment Analysis: {methods_to_run} ===")

    for method in methods_to_run:
        print(f"\n🔄 Running {method.upper()} analysis...")
        
        try:
            sa = SentimentAnalysis(data_path="../result/2.topic_data_stm_WildChat-1M.csv")
            sa.load_data()
            sa.analyze_sentiment_batch(method=method)
            sa.save_results(method=method)
            print(f"✅ {method.upper()} completed")
        except Exception as e:
            print(f"❌ {method.upper()} failed: {str(e)}")
            logger.error(f"Failed {method}: {str(e)}")
            continue

    print(f"\n🎉 Analysis completed! Results: ../result/3.sentiment_data_all_methods.csv")


if __name__ == "__main__":
    main()