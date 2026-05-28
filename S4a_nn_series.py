import pandas as pd
import argparse


print("S4a nn_series started!")

parser = argparse.ArgumentParser()
parser.add_argument("--data_seed", type=int, default=42)
parser.add_argument("--input_folder", type=str, default=None)
args = parser.parse_args()
if args.input_folder is None:
    args.input_folder = f"~/000_data/taiwan/original_100pct_seed{args.data_seed}/"

input_folder = args.input_folder

df = pd.read_feather(f'{input_folder}/nn_series.feather')
df.head(1).to_feather(f'{input_folder}/nn_series__sample.feather')

print("S4a nn_series done!")
