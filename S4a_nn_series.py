import pandas as pd


print("S4a nn_series started!")

input_folder = '~/000_data/taiwan/original_100pct/'

df = pd.read_feather(f'{input_folder}/nn_series.feather')
df.head(1).to_feather(f'{input_folder}/nn_series__sample.feather')

print("S4a nn_series done!")
