import pandas as pd
import statsmodels.formula.api as smf

def run_logit_analysis(social_group: str, sentiment_classifier: str, type: str):
    path = './data/social_groups_with_category_control_sentiment.csv'
    df = pd.read_csv(path)

    # 根据社会群体筛选
    df = df[df['Category'] == social_group]
    df = df[df['sentiment_aliyun'].isin({'positive', 'negative', 'neutral'})]

    if type == 'in':
        df['sentiment_bin'] = df[f'sentiment_{sentiment_classifier}'].apply(lambda x: 1 if x == 'positive' else 0)
        df['group_bin'] = pd.Categorical(df['group'], categories=["they"] + [x for x in df['group'].unique() if x != "they"], ordered=True)
    elif type == 'out':
        df['sentiment_bin'] = df[f'sentiment_{sentiment_classifier}'].apply(lambda x: 1 if x == 'negative' else 0)
        df['group_bin'] = pd.Categorical(df['group'], categories=["we"] + [x for x in df['group'].unique() if x != "we"], ordered=True)

    # 拟合模型
    try:
        formula = 'sentiment_bin ~ C(group_bin) + TTR + TotalTokenScaled'
        model = smf.logit(formula, data=df)
        result = model.fit(disp=False)

        params = result.params.filter(like='C(group_bin)')
        coef = params.values[0]
        std = result.bse[params.index].values[0]
        pval = result.pvalues[params.index].values[0]
        return {
            'SocialGroup': social_group,
            'SentimentClassifier': sentiment_classifier,
            'Coefficient': coef,
            'Std': std,
            'PValue': pval,
            'Type': type
        }
    except Exception as e:
        print(f"Error processing {social_group} {sentiment_classifier} ({type}): {e}")
        return None  # 返回 None 表示跳过该组

def batch_logit_analysis(social_groups_list, sentiment_list, output_csv=''):
    results = []
    for group in social_groups_list:
        for sentiment in sentiment_list:
            result = run_logit_analysis(group, sentiment, type='in')
            if result is not None:
                results.append(result)
                print(f"{group} {sentiment} (in) OK")
    
    for group in social_groups_list:
        for sentiment in sentiment_list:
            result = run_logit_analysis(group, sentiment, type='out')
            if result is not None:
                results.append(result)
                print(f"{group} {sentiment} (out) OK")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# 读取社会群体列表
social_groups_data = pd.read_csv('data/chinese_social_groups.csv')
social_groups_list = social_groups_data['Category'].unique()
sentiment_list = ['aliyun']

# 执行批量回归分析
batch_logit_analysis(social_groups_list, sentiment_list, './data/logistic_result_social_group.csv')
