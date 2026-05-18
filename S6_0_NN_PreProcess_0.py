import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
from tqdm import tqdm

from utils import *
from model import *
root = args.root
seed = args.seed

nn_config = {
    'id_name':id_name,
    'feature_name':[],
    'label_name':label_name,
    'obj_max': 1,
    'epochs': 10,
    'smoothing': 0.001,
    'clipnorm': 1,
    'patience': 100,
    'lr': 3e-4,
    'batch_size': 256,
    'folds': 5,
    'seed': seed,
    'remark': args.remark
}

print("S6_0_NN_PreProcess_0 started...")

input_folder = '~/000_data/taiwan/original_100pct/'

df_full =  pd.read_feather(f'{input_folder}/nn_series.feather')
df_full['idx'] = df_full.index
y = pd.read_csv(f'{input_folder}/train_labels.csv')

# train_df_count_df = pd.read_feather(f'{input_folder}/train_df_count_df.feather')
# train_df_count    = train_df_count_df.values[0][0]

# train_df_count = df_full.groupby('customer_ID',sort=False).idx.agg(['min','max']).reset_index(drop=True).iloc[y.shape[0]-1]['max']

# df_nn_series_train = df_full[:train_df_count+1].reset_index(drop=True)
# df_nn_series_test = df_full[train_df_count+1:].reset_index(drop=True)

df_nn_series_train = pd.read_feather(f'{input_folder}/train_series.feather')
df_nn_series_test  = pd.read_feather(f'{input_folder}/test_series.feather')

df_nn_series_train['idx'] = df_nn_series_train.index
df_nn_series_test['idx'] = df_nn_series_test.index


df_nn_series_idx_train = df_nn_series_train.groupby('customer_ID',sort=False).idx.agg(['min','max']).reset_index(drop=True)
df_nn_series_idx_train['feature_idx'] = np.arange(len(df_nn_series_idx_train))
df_nn_series_train = df_nn_series_train.drop(['idx'],axis=1)

df_nn_series_idx_test = df_nn_series_test.groupby('customer_ID',sort=False).idx.agg(['min','max']).reset_index(drop=True)
df_nn_series_idx_test['feature_idx'] = np.arange(len(df_nn_series_idx_test))
df_nn_series_test = df_nn_series_test.drop(['idx'],axis=1)

df_nn_series_train    .to_feather(f'{input_folder}/df_nn_series_train.feather')
df_nn_series_idx_train.to_feather(f'{input_folder}/df_nn_series_idx_train.feather')

df_nn_series_test     .to_feather(f'{input_folder}/df_nn_series_test.feather')
df_nn_series_idx_test .to_feather(f'{input_folder}/df_nn_series_idx_test.feather')


print("S6_0_NN_PreProcess_0 done...")
