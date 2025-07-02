import pandas as pd
import statsmodels.formula.api as smf

def run_logit_analysis(llm: str, sentiment_classifier: str, type: str):
    path = f'.\\data\\all_data_berttopic_stm_control_sentiment.csv'
    df = pd.read_csv(path)
    df = df[df['model']==llm]
    df = df[df['berttopic'].notnull()]
    df = df[df['sentiment_aliyun'].isin({'positive', 'negative', 'neutral'})]

    if type == 'in':
        df['sentiment_bin'] = df[f'sentiment_{sentiment_classifier}'].apply(lambda x: 1 if x == 'positive' else 0)
        df['group_bin'] = pd.Categorical(df['group'], categories=["they"] + [x for x in df['group'].unique() if x != "they"], ordered=True)
    if type == 'out':
        df['sentiment_bin'] = df[f'sentiment_{sentiment_classifier}'].apply(lambda x: 1 if x == 'negative' else 0)
        df['group_bin'] = pd.Categorical(df['group'], categories=["we"] + [x for x in df['group'].unique() if x != "we"], ordered=True)
    
    df['topic_cate'] = pd.Categorical(df['berttopic'], categories=[-1] + [x for x in df['berttopic'].unique() if x != -1], ordered=True)
    # df['topic_cate'] = pd.Categorical(df['berttopic'], categories=[-1] + [x for x in df['berttopic'].unique() if x != -1], ordered=True)


    # formula = 'sentiment_bin ~ C(group_bin) + TTR + TotalTokenScaled + C(topic_cate)'
    formula = 'sentiment_bin ~ C(group_bin) + TTR + TotalTokenScaled'

    model = smf.logit(formula, data=df)
    result = model.fit(disp=False)
    # print(result.summary())

    params = result.params.filter(like='C(group_bin)')
    coef = params.values[0]
    std = result.bse[params.index].values[0]
    pval = result.pvalues[params.index].values[0]
    return {'LLM': llm, 'SentimentClassifier': sentiment_classifier, 'Coefficient': coef, 'Std': std, 'PValue': pval, 'type': type}


    # coef = result.params['group_bin']
    # pval = result.pvalues['group_bin']

    # return {'LLM': llm, 'SentimentClassifier': sentiment_classifier, 'Coefficient': coef, 'PValue': pval}
    # except Exception as e:
    #     print(f"Error processing {llm} + {sentiment_classifier}: {e}")
    #     return {'LLM': llm, 'SentimentClassifier': sentiment_classifier, 'Coefficient': None, 'PValue': None}

def batch_logit_analysis(llm_list, sentiment_list, output_csv=''):
    results = []
    for llm in llm_list:
        for sentiment in sentiment_list:
            result = run_logit_analysis(llm, sentiment, type='in')
            results.append(result)
            print(llm+' '+sentiment+' OK')
    for llm in llm_list:
        for sentiment in sentiment_list:
            result = run_logit_analysis(llm, sentiment, type='out')
            results.append(result)
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


llm_list = ['Qwen3-8B-Base', 'Qwen-7B', 'Baichuan2-7B-Base', 'glm-4-9b-hf', 'Yi-1.5-6B']
# sentiment_list = ['aliyun', 'tencent', 'erlangshen']
sentiment_list = ['cemotion', 'erlangshen', 'aliyun']


batch_logit_analysis(llm_list, sentiment_list, '.\\data\\logistic_result.csv')