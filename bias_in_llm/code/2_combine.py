import pandas as pd

base_dir='./data/'

models = ['Qwen3-8B-Base']

we_files = ['we_are', 'we_typically', 'we_often', 'we_believe']
they_files = ['they_are', 'they_typically', 'they_often', 'they_believe']

df_combined = []

for model in models:
    save_directory=base_dir + model + '/'
    for i in range(4):
        we_file = we_files[i]
        they_file = they_files[i]

        df_we = pd.read_csv(save_directory+we_file+'_filtered_sentences.csv', encoding='utf-8-sig')
        df_they = pd.read_csv(save_directory+they_file+'_filtered_sentences.csv', encoding='utf-8-sig')
        
        # add group column
        df_we['group'] = 'we'
        df_they['group'] = 'they'

        df_we['source'] = we_file
        df_they['source'] = they_file

        df_we['model'] = model
        df_they['model'] = model

        # conbine two DataFrame
        df_combined.append(df_we)
        df_combined.append(df_they)

result = pd.concat(df_combined, ignore_index=True)

result.to_csv('./data/all_data.csv', index=False, encoding='utf-8-sig')