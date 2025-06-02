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

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Try to import cemotion, if not available, it will be handled gracefully
from cemotion import Cemotion


class SentimentAnalysis:
    def __init__(self, data_path="../result/2.topic_data.csv"):
        """
        Initialize sentiment analysis class with multiple methods

        Supported methods:
        - 'bert': HuggingFace BERT model (Chinese) - auto-detects 2-class or 3-class
        - 'baidu': Baidu AI API (3-class)
        - 'openai': OpenAI/ChatGPT API (configurable)
        - 'vader': VADER-like simple rule-based (continuous)
        - 'cemotion': Cemotion Chinese emotion analysis (2-class)
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
            'bert': {
                'model_name': "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment",
                'cache_dir': os.path.join(hf_cache_dir, 'Models'),
                'confidence_threshold': 0.6,  # For neutral handling in 2-class models
                'auto_detect_classes': True
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
            'vader': {
                'threshold': 0.05  # R-style threshold ±0.05
            },
            'cemotion': {
                'threshold': 0.5,  # 0-0.5: negative, 0.5-1: positive
                'use_binary': True  # Always binary classification for cemotion
            }
        }

    def load_data(self):
        """Load data with topic information"""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path)
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

        if method == 'bert':
            print(f"Loading BERT model: {self.method_configs['bert']['model_name']}")
            print(f"Cache directory: {self.method_configs['bert']['cache_dir']}")

            try:
                tokenizer = BertTokenizer.from_pretrained(
                    self.method_configs['bert']['model_name'],
                    cache_dir=self.method_configs['bert']['cache_dir']
                )
                model = BertForSequenceClassification.from_pretrained(
                    self.method_configs['bert']['model_name'],
                    cache_dir=self.method_configs['bert']['cache_dir']
                )

                # Auto-detect number of classes
                num_classes = model.config.num_labels
                print(f"Model detected: {num_classes}-class classification")

                self.methods[method] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'num_classes': num_classes,
                    'model_name': self.method_configs['bert']['model_name']
                }

                # Store model info for file naming
                self.current_model_info[method] = {
                    'model_name': self.method_configs['bert']['model_name'],
                    'num_classes': num_classes
                }

                print("BERT model loaded successfully")
            except Exception as e:
                print(f"Failed to load BERT model: {e}")
                raise

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
            try:
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
            except Exception as e:
                print(f"Failed to initialize OpenAI API: {e}")
                raise

        elif method == 'vader':
            print("Initializing VADER-like method...")
            positive_words = ['好', '棒', '优秀', '满意', '开心', '高兴', '喜欢', '爱', '完美', '很好']
            negative_words = ['坏', '差', '糟糕', '不满', '生气', '愤怒', '讨厌', '恨', '失望', '不好']
            self.methods[method] = {
                'positive_words': positive_words,
                'negative_words': negative_words
            }

            # Store model info for file naming
            self.current_model_info[method] = {
                'model_name': 'vader-like-chinese',
                'threshold': self.method_configs['vader']['threshold']
            }

            print("VADER-like method initialized successfully")

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

        return self.methods[method]

    def _get_model_identifier(self, method):
        """Generate model identifier for file naming"""
        if method not in self.current_model_info:
            return method

        info = self.current_model_info[method]

        if method == 'bert':
            # Extract simplified model name
            model_name = info['model_name'].split('/')[-1]  # Get last part after /
            return f"bert_{model_name}_{info['num_classes']}class"

        elif method == 'openai':
            model_name = info['model_name'].replace('-', '_')
            class_suffix = "3class" if info['use_three_class'] else "2class"
            return f"openai_{model_name}_{class_suffix}"

        elif method == 'baidu':
            return f"baidu_{info['model_name']}"

        elif method == 'vader':
            threshold_str = str(info['threshold']).replace('.', '_')
            return f"vader_threshold_{threshold_str}"

        elif method == 'cemotion':
            threshold_str = str(info['threshold']).replace('.', '_')
            return f"cemotion_{info['model_name']}_threshold_{threshold_str}"

        else:
            return method

    def _handle_neutral_by_confidence(self, predicted_class, scores, confidence_threshold):
        """Handle neutral sentiment for 2-class models using confidence threshold"""
        max_confidence = max(scores)

        if max_confidence < confidence_threshold:
            # Low confidence -> treat as neutral (R-style: pos=0, neg=0)
            return 0, 0
        else:
            # High confidence -> use predicted class
            pos_score = 1 if predicted_class == 1 else 0
            neg_score = 1 if predicted_class == 0 else 0
            return pos_score, neg_score

    def _handle_three_class_output(self, predicted_class, scores):
        """Handle 3-class model output (R-style: only explicit sentiment gets 1)"""
        # Common 3-class arrangements:
        # Case 1: [negative, neutral, positive] (most common)
        # Case 2: [negative, positive, neutral] (less common)

        # Try to detect arrangement by checking which seems most reasonable
        # For now, assume [negative, neutral, positive] which is most common
        if len(scores) == 3:
            if predicted_class == 0:  # negative
                return 0, 1  # pos=0, neg=1
            elif predicted_class == 2:  # positive
                return 1, 0  # pos=1, neg=0
            else:  # neutral (class 1)
                return 0, 0  # pos=0, neg=0 (R-style)
        else:
            # Fallback to 2-class handling
            pos_score = 1 if predicted_class == 1 else 0
            neg_score = 1 if predicted_class == 0 else 0
            return pos_score, neg_score

    def predict_sentiment_bert(self, text):
        """Predict sentiment using BERT model with dynamic class detection"""
        method_obj = self.init_method('bert')
        model = method_obj['model']
        tokenizer = method_obj['tokenizer']
        num_classes = method_obj['num_classes']

        try:
            inputs = tokenizer(
                text,
                add_special_tokens=True,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                output = model(**inputs)
                predictions = torch.nn.functional.softmax(output.logits, dim=-1)
                scores = predictions[0].tolist()
                predicted_class = torch.argmax(predictions, dim=1).item()

                # Dynamic handling based on number of classes
                if num_classes == 2:
                    # 2-class model: use confidence threshold for neutral (R-style)
                    confidence_threshold = self.method_configs['bert']['confidence_threshold']
                    pos_score, neg_score = self._handle_neutral_by_confidence(
                        predicted_class, scores, confidence_threshold
                    )

                    # Calculate probabilities for 2-class
                    if len(scores) >= 2:
                        positive_prob = scores[1]
                        negative_prob = scores[0]
                    else:
                        positive_prob = scores[0] if predicted_class == 1 else (1 - scores[0])
                        negative_prob = scores[0] if predicted_class == 0 else (1 - scores[0])

                elif num_classes == 3:
                    # 3-class model: direct neutral handling (R-style)
                    pos_score, neg_score = self._handle_three_class_output(predicted_class, scores)

                    # Calculate probabilities for 3-class
                    if len(scores) >= 3:
                        negative_prob = scores[0]  # assuming [neg, neu, pos]
                        positive_prob = scores[2]
                    else:
                        positive_prob = 0.33
                        negative_prob = 0.33

                else:
                    # Fallback for other configurations
                    pos_score = 1 if predicted_class == (num_classes - 1) else 0
                    neg_score = 1 if predicted_class == 0 else 0
                    positive_prob = scores[-1] if len(scores) > 1 else 0.5
                    negative_prob = scores[0] if len(scores) > 1 else 0.5

                return {
                    'pos': pos_score,
                    'neg': neg_score,
                    'positive_prob': positive_prob,
                    'negative_prob': negative_prob,
                    'confidence': max(scores),
                    'raw_scores': scores,
                    'num_classes': num_classes,
                    'predicted_class': predicted_class
                }
        except Exception as e:
            print(f"BERT prediction error: {e}")
            print(f"Text length: {len(text)}")
            return {'pos': 0, 'neg': 0, 'positive_prob': 0.5, 'negative_prob': 0.5, 'confidence': 0.5}

    def predict_sentiment_baidu(self, text):
        """Predict sentiment using Baidu API (3-class with R-style neutral handling)"""
        method_obj = self.init_method('baidu')
        api_url = method_obj['api_url']

        # Truncate text
        encoded = text.encode('utf-8')[:2048]
        text = encoded.decode('utf-8', errors='ignore')

        payload = json.dumps({"text": text}, ensure_ascii=False)
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    data=payload.encode("UTF-8"),
                    verify=False,
                    timeout=30
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    return {'pos': 0, 'neg': 0, 'positive_prob': 0.5, 'negative_prob': 0.5, 'confidence': 0.5}
                time.sleep(2 ** attempt)

        try:
            result = response.json()

            if 'items' in result and len(result['items']) > 0:
                item = result['items'][0]

                sentiment = item.get('sentiment', 1)  # 0:negative, 1:neutral, 2:positive
                confidence = item.get('confidence', 0.5)
                positive_prob = item.get('positive_prob', 0.5)
                negative_prob = item.get('negative_prob', 0.5)

                # R-style binary conversion: only explicit sentiment gets 1
                pos_score = 1 if sentiment == 2 else 0  # only positive(2) -> pos=1
                neg_score = 1 if sentiment == 0 else 0  # only negative(0) -> neg=1
                # neutral(1) -> pos=0, neg=0

                return {
                    'pos': pos_score,
                    'neg': neg_score,
                    'positive_prob': positive_prob,
                    'negative_prob': negative_prob,
                    'confidence': confidence
                }
            else:
                return {'pos': 0, 'neg': 0, 'positive_prob': 0.5, 'negative_prob': 0.5, 'confidence': 0.5}

        except Exception as e:
            return {'pos': 0, 'neg': 0, 'positive_prob': 0.5, 'negative_prob': 0.5, 'confidence': 0.5}

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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.method_configs['openai']['model'],
                    messages=messages
                )

                result_text = response.choices[0].message.content.strip()
                return self._parse_openai_response(result_text, use_three_class)

            except Exception as e:
                if attempt == max_retries - 1:
                    return {'pos': 0, 'neg': 1, 'positive_prob': 0.0, 'negative_prob': 1.0, 'confidence': 1.0}
                time.sleep(2 ** attempt)

    def _parse_openai_response(self, result_text, use_three_class):
        """Parse OpenAI response with dynamic class handling"""
        try:
            sentiment_match = re.search(r'sentiment:\s*(\d+)', result_text)
            confidence_match = re.search(r'confidence:\s*([\d.]+)', result_text)

            if sentiment_match and confidence_match:
                sentiment = int(sentiment_match.group(1))
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))

                if use_three_class:
                    # 3-class: 0=negative, 1=neutral, 2=positive
                    if sentiment == 0:  # negative
                        pos_score, neg_score = 0, 1
                        positive_prob, negative_prob = (1 - confidence) / 2, confidence
                    elif sentiment == 2:  # positive
                        pos_score, neg_score = 1, 0
                        positive_prob, negative_prob = confidence, (1 - confidence) / 2
                    else:  # neutral (sentiment == 1)
                        pos_score, neg_score = 0, 0  # R-style neutral
                        positive_prob, negative_prob = 0.5, 0.5
                else:
                    # 2-class: 0=negative, 1=positive
                    if sentiment == 0:  # negative
                        pos_score, neg_score = 0, 1
                        positive_prob, negative_prob = (1 - confidence), confidence
                    else:  # positive
                        pos_score, neg_score = 1, 0
                        positive_prob, negative_prob = confidence, (1 - confidence)

                return {
                    'pos': pos_score,
                    'neg': neg_score,
                    'positive_prob': positive_prob,
                    'negative_prob': negative_prob,
                    'confidence': confidence
                }
            else:
                return self._parse_openai_fallback(result_text, use_three_class)

        except Exception as e:
            return self._parse_openai_fallback(result_text, use_three_class)

    def _parse_openai_fallback(self, result_text, use_three_class):
        """Fallback parsing for OpenAI response"""
        result_lower = result_text.lower()

        if any(word in result_lower for word in ['negative', 'pessimistic', '负面', '消极', '0']):
            sentiment = 0
        elif any(word in result_lower for word in ['positive', 'optimistic', '正面', '积极']) or '2' in result_lower:
            sentiment = 2 if use_three_class else 1
        elif use_three_class and any(word in result_lower for word in ['neutral', 'objective', '中性', '客观', '1']):
            sentiment = 1
        else:
            sentiment = 0  # default negative

        confidence = 0.6

        if use_three_class:
            if sentiment == 0:
                pos_score, neg_score = 0, 1
                positive_prob, negative_prob = 0.2, 0.6
            elif sentiment == 2:
                pos_score, neg_score = 1, 0
                positive_prob, negative_prob = 0.6, 0.2
            else:  # neutral
                pos_score, neg_score = 0, 0
                positive_prob, negative_prob = 0.4, 0.4
        else:
            if sentiment == 0:
                pos_score, neg_score = 0, 1
                positive_prob, negative_prob = 0.4, 0.6
            else:
                pos_score, neg_score = 1, 0
                positive_prob, negative_prob = 0.6, 0.4

        return {
            'pos': pos_score,
            'neg': neg_score,
            'positive_prob': positive_prob,
            'negative_prob': negative_prob,
            'confidence': confidence
        }

    def predict_sentiment_vader(self, text):
        """Predict sentiment using VADER-like method (R-style threshold conversion)"""
        method_obj = self.init_method('vader')
        positive_words = method_obj['positive_words']
        negative_words = method_obj['negative_words']

        # Simple word counting approach
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)

        # Calculate compound score (simulating VADER compound score)
        total_words = len(text.split())
        if total_words > 0:
            score = (pos_count - neg_count) / total_words
        else:
            score = 0.0

        # R-style threshold conversion (threshold ±0.05)
        threshold = self.method_configs['vader']['threshold']
        pos_score = 1 if score > threshold else 0
        neg_score = 1 if score < -threshold else 0
        # Neutral range [-0.05, 0.05]: pos=0, neg=0

        return {
            'pos': pos_score,
            'neg': neg_score,
            'positive_prob': max(0, score),
            'negative_prob': max(0, -score),
            'confidence': abs(score) + 0.5,
            'vader_score': score
        }

    def predict_sentiment_cemotion(self, text):
        """Predict sentiment using Cemotion model (R-style binary conversion)"""
        method_obj = self.init_method('cemotion')
        cemotion_model = method_obj['model']
        threshold = method_obj['threshold']

        try:
            # Get cemotion score (0-1 range)
            score = cemotion_model.predict(text)

            # R-style binary conversion based on threshold
            if score < threshold:
                # Negative sentiment
                pos_score = 0
                neg_score = 1
                predicted_class = 0
                positive_prob = score
                negative_prob = 1 - score
            else:
                # Positive sentiment
                pos_score = 1
                neg_score = 0
                predicted_class = 1
                positive_prob = score
                negative_prob = 1 - score

            return {
                'pos': pos_score,
                'neg': neg_score,
                'positive_prob': positive_prob,
                'negative_prob': negative_prob,
                'confidence': max(positive_prob, negative_prob),
                'cemotion_score': score,
                'predicted_class': predicted_class
            }

        except Exception as e:
            print(f"Cemotion prediction error: {e}")
            return {'pos': 0, 'neg': 0, 'positive_prob': 0.5, 'negative_prob': 0.5, 'confidence': 0.5,
                    'cemotion_score': 0.5}

    def analyze_sentiment_batch(self, method='bert'):
        """
        Batch sentiment analysis with dynamic class detection

        Args:
            method: 'bert', 'baidu', 'openai', 'vader', or 'cemotion'
        """
        if self.df is None:
            self.load_data()

        print(f"Performing sentiment analysis using {method.upper()} method...")

        # Select prediction function
        predict_funcs = {
            'bert': self.predict_sentiment_bert,
            'baidu': self.predict_sentiment_baidu,
            'openai': self.predict_sentiment_openai,
            'vader': self.predict_sentiment_vader,
            'cemotion': self.predict_sentiment_cemotion
        }

        if method not in predict_funcs:
            raise ValueError(f"Unsupported method: {method}. Choose from {list(predict_funcs.keys())}")

        predict_func = predict_funcs[method]

        results = []
        texts = self.df['text'].tolist()

        # Add progress bar
        for i in tqdm(range(len(texts)), desc=f"{method.upper()} sentiment analysis"):
            text = str(texts[i])
            result = predict_func(text)
            results.append(result)

            # Add delay for API methods (not needed for cemotion)
            if method in ['baidu', 'openai']:
                time.sleep(0.5)

        # Unified column names
        self.df['pos'] = [r['pos'] for r in results]
        self.df['neg'] = [r['neg'] for r in results]

        # Probability scores
        self.df['positive_prob'] = [r['positive_prob'] for r in results]
        self.df['negative_prob'] = [r['negative_prob'] for r in results]
        self.df['confidence'] = [r['confidence'] for r in results]

        # Print model info for BERT
        if method == 'bert' and results:
            sample_result = results[0]
            if 'num_classes' in sample_result:
                print(f"Model type: {sample_result['num_classes']}-class classification")

        # Print model info for Cemotion
        if method == 'cemotion' and results:
            print(
                f"Model type: Cemotion binary classification (threshold: {self.method_configs['cemotion']['threshold']})")

        print(f"{method.upper()} sentiment analysis completed")
        return results

    def save_results(self, method, output_dir="../result/"):
        """Save results with model-specific filename"""
        if self.df is None:
            raise ValueError("No data to save")

        # Generate model-specific identifier
        model_identifier = self._get_model_identifier(method)
        output_path = f"{output_dir}3.sentiment_data_{model_identifier}.csv"

        # Add metadata to the dataframe
        metadata_cols = {}
        if method in self.current_model_info:
            info = self.current_model_info[method]
            for key, value in info.items():
                metadata_cols[f'model_{key}'] = value

        # Create a copy with metadata
        df_to_save = self.df.copy()
        for col, value in metadata_cols.items():
            df_to_save[col] = value

        df_to_save.to_csv(output_path, index=False, encoding='utf-8')
        print(f"Results saved to: {output_path}")
        return output_path

    def analyze_sentiment_statistics(self, method='bert'):
        """Analyze sentiment statistics for specific method"""
        stats = {}

        pos_col = 'pos'
        neg_col = 'neg'

        if pos_col not in self.df.columns:
            print(f"Warning: {method} results not found. Run analysis first.")
            return stats

        # Calculate statistics similar to R version
        method_stats = {
            'positive_count': self.df[pos_col].sum(),
            'negative_count': self.df[neg_col].sum(),
            'neutral_count': len(self.df) - self.df[pos_col].sum() - self.df[neg_col].sum(),
            'positive_rate': self.df[pos_col].mean(),
            'negative_rate': self.df[neg_col].mean(),
            'neutral_rate': (len(self.df) - self.df[pos_col].sum() - self.df[neg_col].sum()) / len(self.df)
        }

        # Group statistics if available
        if 'ingroup' in self.df.columns:
            ingroup_stats = self.df[self.df['ingroup'] == 1]
            outgroup_stats = self.df[self.df['ingroup'] == 0]

            method_stats['ingroup'] = {
                'positive_rate': ingroup_stats[pos_col].mean(),
                'negative_rate': ingroup_stats[neg_col].mean(),
                'count': len(ingroup_stats)
            }
            method_stats['outgroup'] = {
                'positive_rate': outgroup_stats[pos_col].mean(),
                'negative_rate': outgroup_stats[neg_col].mean(),
                'count': len(outgroup_stats)
            }

        stats[method] = method_stats

        # Display model information
        if method in self.current_model_info:
            print(f"\n=== {method.upper()} Model Info ===")
            for key, value in self.current_model_info[method].items():
                print(f"{key}: {value}")

        print(f"\n=== {method.upper()} Statistics ===")
        print(f"Positive rate: {method_stats['positive_rate']:.4f}")
        print(f"Negative rate: {method_stats['negative_rate']:.4f}")
        print(f"Neutral rate: {method_stats['neutral_rate']:.4f}")
        if 'ingroup' in method_stats:
            print(f"Ingroup positive rate: {method_stats['ingroup']['positive_rate']:.4f}")
            print(f"Outgroup positive rate: {method_stats['outgroup']['positive_rate']:.4f}")

        return stats

    def _get_baidu_access_token(self):
        """Get Baidu API access token"""
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            "grant_type": "client_credentials",
            "client_id": self.method_configs['baidu']['api_key'],
            "client_secret": self.method_configs['baidu']['secret_key']
        }
        return str(requests.post(url, params=params).json().get("access_token"))

    def get_enhanced_data(self):
        """Return dataframe with sentiment analysis results"""
        return self.df

    def set_model_config(self, method, **kwargs):
        """Update model configuration for a specific method"""
        if method in self.method_configs:
            self.method_configs[method].update(kwargs)
            # Clear cached method to force reinitialization
            if method in self.methods:
                del self.methods[method]
            print(f"Updated {method} configuration: {kwargs}")
        else:
            print(f"Unknown method: {method}")


def main():
    """Main function for testing"""
    methods_to_run = ['cemotion']  # Change this to ['bert', 'baidu', 'openai', 'vader', 'cemotion'] to run all

    for method in methods_to_run:
        print(f"\n{'=' * 50}")
        print(f"Running {method.upper()} sentiment analysis")
        print(f"{'=' * 50}")

        sa = SentimentAnalysis(data_path="../result/2.topic_data.csv")
        try:
            sa.load_data()
            sa.analyze_sentiment_batch(method=method)
            sa.analyze_sentiment_statistics(method=method)
            sa.save_results(method=method)

        except Exception as e:
            print(f"Error with {method}: {e}")

    print("\nAll sentiment analysis completed!")


if __name__ == "__main__":
    main()