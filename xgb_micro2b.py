# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 22:34:03 2024

@author: Zhouyu Shen
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 20:26:26 2024

@author: Zhouyu Shen
"""
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import xgboost as xgb
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from scipy.io import loadmat
from keras.regularizers import l2, l1
from keras.initializers import VarianceScaling
import random
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)


def fit_models(num1, num2):
    np.random.seed(an)
    random.seed(an)    
    tf.random.set_seed(an)  
    
    filepath='weak/Micro2data/'
    data = loadmat(filepath+f"Micro2_{num2+1}_{num1+1}.mat")
    X_train = data['X_train']
    X_pred = data['X_pred']
    y_train = data['y_train']
    y_pred = data['y_pred']
    W_train = data['W_train']
    W_pred = data['W_pred']

    gammahat = np.linalg.solve(W_train.T @ W_train, W_train.T @ y_train)
    y_predb= W_pred @ gammahat
    
    Mw=np.eye(W_train.shape[0]) - np.dot(np.dot(W_train, np.linalg.inv(np.dot(W_train.T, W_train))), W_train.T)
    X_pred=X_pred-W_pred@ np.linalg.inv(W_train.T @ W_train)@ W_train.T @ X_train
    X_train=Mw@X_train
    y_train=Mw@y_train
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train = (X_train - X_mean) / X_std
    X_pred = (X_pred - X_mean) / X_std
    
    param_grid = {
        'max_depth': [1, 2, 3, 4, 5,6,7,8,9,10],
        'learning_rate':  [1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10],
        'max_n_estimators': 50,
        }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_pred, label=y_pred)
    best_score = float("inf")
    best_params = None
    max_n_estimators = param_grid['max_n_estimators']
    
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            params = {
                'objective': 'reg:squarederror',
                'seed': an,
                'reg_alpha': 0,
                'reg_lambda': 0,
                'booster': 'gbtree',
                'n_jobs': -1,
                'max_depth': max_depth, 
                'learning_rate': learning_rate
            }
            

            cv_results = xgb.cv(params, dtrain, num_boost_round=max_n_estimators, nfold=10, metrics='rmse')
            
            min_rmse_index = np.argmin(cv_results['test-rmse-mean'])
            mean_rmse = cv_results['test-rmse-mean'][min_rmse_index]
            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = params.copy()
                best_params['n_estimators'] = min_rmse_index + 1  
    # 使用最佳参数训练模型
    final_model = xgb.train(best_params, dtrain, num_boost_round=best_params['n_estimators'])

    y_predhat = final_model.predict(dtest)
    f=open("weak/micro2b_xgb_result.txt","a")
    f.write(f"{num2} {((y_pred - y_predb-y_predhat) ** 2).mean()} { ((y_pred - y_predb) ** 2).mean()} {best_params['max_depth']} {best_params['learning_rate']} {best_params['n_estimators']}"+'\n')
    return 1

num1 = 0
num2 = 0
results = []
#num1=j num2=i
#an 1-100
for i in range(1, 101):
    if i==an:
        fit_models(num1,num2)
    if(num1 == 19):
      num2+=1
      num1 = 0
    else: 
      num1+=1


'''
file_path = 'micro2b_xgb_result.txt'
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None)
df.columns = ['i', 'nn', 'mean']

final_temp = pd.DataFrame(np.zeros((1, 1)))
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    temp[:-1] = 1 - temp[:-1]/temp[-1]
    final_temp.iloc[0] += temp[:-1].values/5
print(final_temp)

# Assuming you have already read the DataFrame
df = pd.read_csv(file_path, sep=' ', header=None)

# Select the last three columns
df_last_three = df.iloc[:, -3:]

# Create a figure with 3 subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Histogram for the third last column
axes[0].hist(df_last_three.iloc[:, 0], bins=20)
axes[0].set_title('Third Last Column Histogram')

# Histogram for the second last column with log transformation
# Replace zeros with a very small number to avoid log(0)
df_last_three.iloc[:, 1].replace(0, np.finfo(float).eps, inplace=True)
axes[1].hist(np.log10(df_last_three.iloc[:, 1]), bins=100)
axes[1].set_title('Second Last Column Log Transformed Histogram')

# Histogram for the last column
axes[2].hist(df_last_three.iloc[:, 2], bins=20)
axes[2].set_title('Last Column Histogram')

# Display the histograms
plt.tight_layout()
plt.show()
'''