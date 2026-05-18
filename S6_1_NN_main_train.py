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

print("S6_1_NN_main_train started!")

input_folder = args.input_folder

nn_config = {
    'id_name':id_name,
    'feature_name':[],
    'label_name':label_name,
    'obj_max': 1,
    'epochs': args.epochs,  ## 12
    'smoothing': 0.001,
    'clipnorm': 1,
    'patience': 100,
    'lr': 3e-4,
    'batch_size': 512, #256,
    'folds': 5,
    'seed': seed,
    'remark': args.remark
}

df_nn_series_train     = pd.read_feather(f'{input_folder}/df_nn_series_train.feather')
df_nn_series_idx_train = pd.read_feather(f'{input_folder}/df_nn_series_idx_train.feather')
y = pd.read_csv(f'{input_folder}/train_labels.csv')
 

if not args.use_fe:
    df_nn_feature_train = pd.DataFrame([None]*len(y))
    
    NN_train(     [df_nn_series_train,df_nn_feature_train,y,df_nn_series_idx_train.values]
                 ,Amodel
                 ,nn_config
                 ,use_series_oof=False
                 ,run_id='NN_with_series_feature'
                )

else:
    df_nn_feature_train = pd.read_feather(f'{input_folder}/df_nn_feature_train.feather')
    
    NN_train(     [df_nn_series_train,df_nn_feature_train,y,df_nn_series_idx_train.values]
                 ,Amodel
                 ,nn_config
                 ,use_series_oof=True
                 ,run_id='NN_with_series_and_all_feature'
            )


print("S6_1_NN_main_train done!")
