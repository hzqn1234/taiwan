# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable. This study reviewed the literature and used the following 23 variables as explanatory variables:
# X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
# X2: Gender (1 = male; 2 = female).
# X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others).
# X4: Marital status (1 = married; 2 = single; 3 = others).
# X5: Age (year).
# X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: X6 = the repayment status in September, 2005; X7 = the repayment status in August, 2005; . . .;X11 = the repayment status in April, 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
# X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September, 2005; X13 = amount of bill statement in August, 2005; . . .; X17 = amount of bill statement in April, 2005. 
# X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September, 2005; X19 = amount paid in August, 2005; . . .;X23 = amount paid in April, 2005.

# categorical: X2, X3, X4, X5
# time series: X6~X11, X12~X17, X18~X23


import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo 

print("S1 data preparation started!")

# fetch dataset 
default_of_credit_card_clients = fetch_ucirepo(id=350) 
  
# data (as pandas dataframes) 
x = default_of_credit_card_clients.data.features 
y = default_of_credit_card_clients.data.targets 

print(f"check null values: {x.isna().sum().sum()}")

input_folder = '~/000_data/taiwan/original_100pct/'

y = y.reset_index().rename({'index':'customer_ID', 'Y':'target'}, axis=1)
y.to_csv(f'{input_folder}/label.csv', index=False)

train_ratio, test_ratio = 0.7, 0.3
train_list = []
test_list = []

seed = 42
np.random.seed(seed)

for i in range(len(y)):
    rann = np.random.random()
    if rann < train_ratio:
        train_list.append(i)
    else:
        test_list.append(i)

train_index = pd.DataFrame(data=train_list, columns=['customer_ID'])
test_index  = pd.DataFrame(data=test_list, columns=['customer_ID'])

train_y =  y.merge(train_index, on = 'customer_ID')
test_y  =  y.merge(test_index, on = 'customer_ID')

train_y.to_csv(f'{input_folder}/train_labels.csv', index=False)
test_y.to_csv (f'{input_folder}/test_labels.csv', index=False)

x = x.reset_index().rename({'index':'customer_ID'}, axis=1)
x.to_csv(f'{input_folder}/all_feature.csv', index=False)
x.to_feather(f'{input_folder}/all_feature.feather')
x.to_feather(f'{input_folder}/nn_all_feature.feather')

x.head(1).to_feather(f'{input_folder}/all_feature_sample.feather')
x.head(1).to_feather(f'{input_folder}/nn_all_feature_sample.feather')


# X6 -X11
# X12-X17
# X18-X23

m_09 = x[['customer_ID','X6' ,'X12','X18']].rename({'X6' :'repay_status', 'X12':'bill', 'X18':'paid'}, axis=1)
m_08 = x[['customer_ID','X7' ,'X13','X19']].rename({'X7' :'repay_status', 'X13':'bill', 'X19':'paid'}, axis=1)
m_07 = x[['customer_ID','X8' ,'X14','X20']].rename({'X8' :'repay_status', 'X14':'bill', 'X20':'paid'}, axis=1)
m_06 = x[['customer_ID','X9' ,'X15','X21']].rename({'X9' :'repay_status', 'X15':'bill', 'X21':'paid'}, axis=1)
m_05 = x[['customer_ID','X10','X16','X22']].rename({'X10':'repay_status', 'X16':'bill', 'X22':'paid'}, axis=1)
m_04 = x[['customer_ID','X11','X17','X23']].rename({'X11':'repay_status', 'X17':'bill', 'X23':'paid'}, axis=1)

m_09['S2']=9
m_08['S2']=8
m_07['S2']=7
m_06['S2']=6
m_05['S2']=5
m_04['S2']=4

series = pd.concat([m_04,m_05,m_06,m_07,m_08,m_09])
series = series.sort_values(by=['customer_ID','S2'], ascending=True).reset_index(drop=True)

series.to_csv(f'{input_folder}/nn_series.csv', index=False)
series.to_feather(f'{input_folder}/nn_series.feather')
series.head(1).to_feather(f'{input_folder}/nn_series__sample.feather')

print(f"series null check: {series.isnull().sum()}")

train_series =  series.merge(train_index, on = 'customer_ID')
test_series  =  series.merge(test_index, on = 'customer_ID')

train_series.to_csv(f'{input_folder}/train_series.csv', index=False)
train_series.to_feather(f'{input_folder}/train_series.feather')

test_series.to_csv (f'{input_folder}/test_series.csv', index=False)
test_series.to_feather(f'{input_folder}/test_series.feather')

train_x =  x.merge(train_index, on = 'customer_ID')
test_x  =  x.merge(test_index, on = 'customer_ID')

train_x.to_csv(f'{input_folder}/train.csv', index=False)
train_x.to_feather(f'{input_folder}/train.feather')

test_x.to_csv (f'{input_folder}/test.csv', index=False)
test_x.to_feather(f'{input_folder}/test.feather')

print(f"train_x len: {len(train_x)}, test_x len: {len(test_x)}")

print("S1 data preparation done!")
