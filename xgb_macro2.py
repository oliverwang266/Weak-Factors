# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:09:04 2023

@author: Zhouyu Shen
"""
'''
import os
os.chdir('C:\\Users\Zhouyu Shen\Dropbox\PC\Desktop/research/weak signal/Can machines learn weak signals/code/realdata_jie/real/real/nn_empirical')
'''
# Micro1_scale.txt
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import random
import math
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

Macro2 = loadmat("weak/GrowthData.mat")
data = Macro2["data"]
data = np.column_stack((np.ones(data.shape[0]), data))


p = data.shape[1] - 1
n = data.shape[0]
y = data[:, p]
X = data[:, :p]
X = X[:, 1:]
p = X.shape[1]

# Generate a list of n_estimators values from p/10 to p, stepping by p/10

def fit_models(X, y, index_train, index_pred):
    np.random.seed(an)
    random.seed(an)
    y_train = 10*y[index_train]
    X_train = X[index_train, :]
    y_pred = 10*y[index_pred]
    X_test = X[index_pred, :]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    dtrain = xgb.DMatrix(X_train, label=y_train-y_mean)
    dtest = xgb.DMatrix(X_test, label=y_pred-y_mean)
    
    param_grid = {
        'max_depth': [1, 2, 3, 4, 5,6,7,8,9,10],
        'learning_rate': [0.5, 0.2, 0.1,0.05,0.02,0.01],
        'max_n_estimators': 500,
        }

    
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

    f=open("weak/macro2_xgb_result.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()/100} { ((y_pred - y_mean) ** 2).mean()/100} {best_params['max_depth']} {best_params['learning_rate']} {best_params['n_estimators']}"+'\n')

    return 1
#an:1-100
results = []
f = open("weak/Macro2_index.txt", "r")
lines = f.readlines()
lines1 = lines[an][:-1]
line = np.array(lines1.split(' '))
# find the index where the value is -1000
train_endindex = np.where(line == '-1000')[0][0]
train_index = line[2:train_endindex].astype(int) -1
test_endindex = np.where(line == '-2000')[0][0]
test_index = line[train_endindex+1:test_endindex].astype(int) -1
final, tuning = fit_models(X, y, train_index, test_index)



'''
file_path = 'macro2_xgb_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None)
column_means = df.mean()
print(column_means)
col_1 = df.iloc[:, 1]
col_2 = df.iloc[:, 2]

result = 1 - (col_1 / col_2)

mean_value = result.mean()

print(mean_value)
# Plot histograms for the last three columns
plt.figure(figsize=(15, 5))

# Histogram for the last column
plt.subplot(1, 3, 1)
plt.hist(df.iloc[:, -1], bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Last Column')

# Histogram for the second-to-last column (logarithmic)
plt.subplot(1, 3, 2)
plt.hist(np.log10(df.iloc[:, -2]), bins=20, color='green', edgecolor='black')
plt.title('Log Histogram of Second-to-Last Column')

# Histogram for the third-to-last column
plt.subplot(1, 3, 3)
plt.hist(df.iloc[:, -3], bins=20, color='red', edgecolor='black')
plt.title('Histogram of Third-to-Last Column')

plt.tight_layout()
plt.show()

'''
