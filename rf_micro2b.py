# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 22:59:18 2024

@author: Oliver Wang
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
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
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
    
    rf_grid = {
        'max_depth': np.arange(1, 6, 1),
        'max_features':  np.arange(1, 6, 1)
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rfr = RandomForestRegressor(n_estimators=500, random_state=an)
    grid = GridSearchCV(estimator=rfr,scoring=scorer, param_grid=rf_grid, cv=10,n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    y_predhat = grid_result.best_estimator_.predict(X_pred)
 
    f=open("weak/micro2b_rf_result.txt","a")
    f.write(f"{num2} {((y_pred- y_predb-y_predhat) ** 2).mean()} { ((y_pred - y_predb) ** 2).mean()} {grid_result.best_params_['max_depth']} {grid_result.best_params_['max_features']}"+'\n')
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
file_path = 'micro2b_rf_result.txt'
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None)
df.columns = ['i', 'nn', 'mean']

final_temp = pd.DataFrame(np.zeros((1, 1)))
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    temp[:-1] = 1 - temp[:-1]/temp[-1]
    final_temp.iloc[0] += temp[:-1].values/5
print(final_temp)

df = pd.read_csv(file_path, sep='\s+',  header=None)
# Assuming df is already loaded with your data
col_1 = df.iloc[:, -2]  # Second last column
col_2 = df.iloc[:, -1]  # Last column

# Creating a figure with 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Plotting histogram for second last column
axes[0].hist(col_1, bins=40, color='blue', alpha=0.7)
axes[0].set_title('Histogram of Second Last Column')
axes[0].set_xlabel('Values')
axes[0].set_ylabel('Frequency')

# Plotting histogram for last column
axes[1].hist(col_2, bins=40, color='green', alpha=0.7)
axes[1].set_title('Histogram of Last Column')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')

# Display the plot
plt.tight_layout()
plt.show()
'''
