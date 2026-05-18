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
    'epochs': 12,
    'smoothing': 0.001,
    'clipnorm': 1,
    'patience': 100,
    'lr': 3e-4,
    'batch_size': 256,
    'folds': 5,
    'seed': seed,
    'remark': args.remark
}

df_nn_series_test     = pd.read_feather('./input/df_nn_series_test.feather')
df_nn_series_idx_test = pd.read_feather('./input/df_nn_series_idx_test.feather')
y = pd.read_csv('./input/train_labels.csv')
 

if not args.use_fe:
    df_nn_feature_test = df_nn_series_test[['customer_ID']].drop_duplicates().reset_index(drop=True)
    # df_nn_feature_test = pd.DataFrame([None]*len(y))
    # df_nn_feature_test = pd.read_feather('./input/df_nn_feature_test.feather')
    
    NN_predict(     [df_nn_series_test,df_nn_feature_test,df_nn_series_idx_test.values]
                 ,Amodel
                 ,nn_config
                 ,use_series_oof=False
                 ,run_id='NN_with_series_feature'
                )

else:
    df_nn_feature_test = pd.read_feather('./input/df_nn_feature_test.feather')
    
    NN_predict(     [df_nn_series_test,df_nn_feature_test,df_nn_series_idx_test.values]
                 ,Amodel
                 ,nn_config
                 ,use_series_oof=True
                 ,run_id='NN_with_series_and_all_feature'
            )


# df =  pd.read_feather('./input/nn_series.feather')
# y = pd.read_csv('./input/train_labels.csv')

# f = pd.read_feather('./input/nn_all_feature.feather')
# df['idx'] = df.index
# series_idx = df.groupby('customer_ID',sort=False).idx.agg(['min','max'])
# series_idx['feature_idx'] = np.arange(len(series_idx))
# df = df.drop(['idx'],axis=1)
# print(f.head())

