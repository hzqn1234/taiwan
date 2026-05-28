#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gc,os,random
import time,datetime
import argparse
from tqdm import tqdm

print("S8_check_submission started!")

parser = argparse.ArgumentParser()
parser.add_argument("--data_seed", type=int, default=42)
parser.add_argument("--input_folder", type=str, default=None)
args = parser.parse_args()
if args.input_folder is None:
    args.input_folder = f"~/000_data/taiwan/original_100pct_seed{args.data_seed}/"

input_folder = args.input_folder

p0 = pd.read_csv('./output/LGB_with_manual_feature/submission.csv.zip')
# p1 = pd.read_csv('./output/LGB_with_manual_feature_and_series_oof/submission.csv.zip')
# p2 = pd.read_csv('./output/NN_with_series_feature/submission.csv.zip')
p3 = pd.read_csv('./output/NN_with_series_and_all_feature/submission.csv.zip')

# p0['prediction'] = p0['prediction']*0.3 + p1['prediction']*0.35 + p2['prediction']*0.15 + p3['prediction']*0.1

# p0.to_csv('./output/final_submission.csv.zip',index=False, compression='zip')

p_ensemble = p0[['customer_ID']]
p_ensemble['prediction'] = p0['prediction']*0.5 + p3['prediction']*0.5

test_label = pd.read_csv(f'{input_folder}/test_labels.csv')

from sklearn.metrics import roc_auc_score, f1_score

def amex_metric_mod(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

def Metric(labels,preds):
    return amex_metric_mod(labels,preds)


print('Amex Metric:')
print('LGBM: ', Metric(test_label['target'].values, p0['prediction'].values))
print('NN: ', Metric(test_label['target'].values, p3['prediction'].values))
print('ensemble: ', Metric(test_label['target'].values, p_ensemble['prediction'].values))


print('F1:')

results_list = []
for t0 in range(1,10):
    t=t0/10
    print(f't: {t}')
    results_list.append({
        't': f'{t}',
        'F1_LGB': round(f1_score(test_label['target'].values, p0['prediction'].values >= t),3),
        'F1_NN': round(f1_score(test_label['target'].values, p3['prediction'].values >= t),3),
        'F1_Ensemble': round(f1_score(test_label['target'].values, p_ensemble['prediction'].values >= t),3),
    })
    
    # print('LGBM: ', f1_score(test_label['target'].values, p0['prediction'].values >= t))
    # print('NN: ', f1_score(test_label['target'].values, p3['prediction'].values >= t))
    # print('ensemble: ', f1_score(test_label['target'].values, p_ensemble['prediction'].values >= t))

results_df = pd.DataFrame(results_list)
results_df.to_csv(f'./output/f1_dict.csv', index=False)


print("S8_check_submission done!")
