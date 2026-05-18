import pandas as pd

print("S4a feature sample started!")

input_folder = '~/000_data/taiwan/original_100pct/'

df = pd.read_feather(f'{input_folder}/all_feature.feather')
df.head(1).to_feather(f'{input_folder}/all_feature_sample.feather')

del df

df = pd.read_feather(f'{input_folder}/nn_all_feature.feather')
df.head(1).to_feather(f'{input_folder}/nn_all_feature_sample.feather')

del df

train = pd.read_feather(f'{input_folder}/train.feather')
train_df_count_df = pd.DataFrame([train.shape[0]],columns=['train_df_count'])
train_df_count_df.to_feather(f'{input_folder}/train_df_count_df.feather')

print("S4a feature sample done!")
