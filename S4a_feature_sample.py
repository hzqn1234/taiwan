import pandas as pd
import argparse

print("S4a feature sample started!")

parser = argparse.ArgumentParser()
parser.add_argument("--data_seed", type=int, default=42)
parser.add_argument("--input_folder", type=str, default=None)
args = parser.parse_args()
if args.input_folder is None:
    args.input_folder = f"~/000_data/taiwan/original_100pct_seed{args.data_seed}/"

input_folder = args.input_folder

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
