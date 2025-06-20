import pandas as pd
import statsmodels.formula.api as smf

def run_logit_analysis(llm: str, sentiment_classifier: str, type: str):
    path = f'.\\data\\all_data_berttopic_stm_control_sentiment.csv'
    df = pd.read_csv(path)
    df = df[df['dominant_topic'].notnull()]

    if type == 'in':
        df['sentiment_bin'] = df[f'sentiment_{sentiment_classifier}'].apply(lambda x: 1 if x == 'positive' else 0)
        df['group_bin'] = pd.Categorical(df['group'], categories=["they"] + [x for x in df['group'].unique() if x != "they"], ordered=True)
    if type == 'out':
        df['sentiment_bin'] = df[f'sentiment_{sentiment_classifier}'].apply(lambda x: 1 if x == 'negative' else 0)
        df['group_bin'] = pd.Categorical(df['group'], categories=["we"] + [x for x in df['group'].unique() if x != "we"], ordered=True)
    
    df['topic_cate'] = pd.Categorical(df['dominant_topic'], categories=[-1] + [x for x in df['dominant_topic'].unique() if x != -1], ordered=True)
    # df['topic_cate'] = pd.Categorical(df['berttopic'], categories=[-1] + [x for x in df['berttopic'].unique() if x != -1], ordered=True)


    formula = 'sentiment_bin ~ C(group_bin) + TTR + TotalTokenScaled + C(topic_cate)'
    # formula = 'sentiment_bin ~ C(group_bin) + TTR + TotalTokenScaled'

    model = smf.logit(formula, data=df)
    result = model.fit(disp=False)
    print(result.summary())

    # coef = result.params['group_bin']
    # pval = result.pvalues['group_bin']

    # return {'LLM': llm, 'SentimentClassifier': sentiment_classifier, 'Coefficient': coef, 'PValue': pval}
    # except Exception as e:
    #     print(f"Error processing {llm} + {sentiment_classifier}: {e}")
    #     return {'LLM': llm, 'SentimentClassifier': sentiment_classifier, 'Coefficient': None, 'PValue': None}

def batch_logit_analysis(llm_list, sentiment_list, type, output_csv=''):
    results = []
    for llm in llm_list:
        for sentiment in sentiment_list:
            result = run_logit_analysis(llm, sentiment, type)
            results.append(result)

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


llm_list = ['Qwen3-8B-Base']
# sentiment_list = ['aliyun', 'tencent', 'erlangshen']
sentiment_list = ['erlangshen']


batch_logit_analysis(llm_list, sentiment_list, 'out', '.\\data\\logistic_result_in.csv')