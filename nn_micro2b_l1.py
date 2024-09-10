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

import numpy as np
import pandas as pd
from scipy.io import loadmat
from keras.regularizers import l2, l1
from keras.initializers import VarianceScaling
import random
from sklearn.metrics import make_scorer
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

filepath='weak/Micro2data/'
data = loadmat(filepath+f"Micro2_1_1.mat")
X= data['X_train']


def create_model(hidden_units=[16], lambda_value=10, epochs=100, init_var=0.01):
    np.random.seed(an)
    random.seed(an)    
    tf.random.set_seed(an)  
    lr=.5/epochs
    def custom_initializer():
        return VarianceScaling(scale=init_var)
    model = Sequential()

    # Add the first layer with the specified input_dim
    model.add(Dense(hidden_units[0], input_dim=X.shape[1], kernel_regularizer=l1(lambda_value), 
                    activation='relu',kernel_initializer=custom_initializer()))

    # If there are more hidden layers, add them sequentially
    for units in hidden_units[1:]:
        model.add(Dense(units, kernel_regularizer=l1(lambda_value), 
                        activation='relu',kernel_initializer=custom_initializer()))

    model.add(Dense(1, kernel_regularizer=l1(lambda_value), use_bias=False,
                    activation='linear',kernel_initializer=custom_initializer()))
    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=lr))
    return model

def fit_models(num1, num2):
    np.random.seed(an)
    random.seed(an)    
    tf.random.set_seed(an)  
    
    filepath='weak/Micro2data/'
    data = loadmat(filepath+f"Micro2_{num2+1}_{num1+1}.mat")
    X_train = data['X_train']
    X_pred = data['X_pred']
    y_train =data['y_train']
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
    
    model = KerasRegressor(build_fn=create_model, batch_size=16, verbose=0)

    nn_grid = {
        #'lambda_value': np.arange(0.0095, 0.0105, 0.0005),
        'lambda_value': 10**np.arange(0,2+0.1, .2),
        'epochs': [50,100,200]
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid = GridSearchCV(estimator=model, param_grid=nn_grid, cv=10,scoring=scorer, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    # Predict on the test set
    y_predhat = grid_result.best_estimator_.predict(X_pred,verbose=0)
    f=open("weak/micro2b_l1_result.txt","a")
    f.write(f"{an} {num2} {((y_pred - y_predb-y_predhat) ** 2).mean()} { ((y_pred - y_predb) ** 2).mean()} {grid_result.best_params_['epochs']} {grid_result.best_params_['lambda_value']}"+'\n')
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
file_path = 'micro2b_l1_result.txt'
df = pd.read_csv(file_path, sep='\s+', usecols=[1, 2, 3], header=None)
df.columns = ['i', 'nn', 'mean']

final_temp = pd.DataFrame(np.zeros((1, 1)))
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    temp[:-1] = 1 - temp[:-1]/temp[-1]
    final_temp.iloc[0] += temp[:-1].values/5
print(final_temp)


df = pd.read_csv(file_path, sep='\s+', header=None)

# Check for non-positive values in the last column to avoid log errors
if (df.iloc[:, -1] <= 0).any():
    print("Warning: Non-positive values found in the last column, which may cause errors with logarithm transformation.")

# Applying log transformation
df.iloc[:, -1] = np.log10(df.iloc[:, -1])

# Plotting histograms
plt.figure(figsize=(12, 5))

# Histogram for the second-to-last column
plt.subplot(1, 2, 1)
plt.hist(df.iloc[:, -2], bins=20, edgecolor='black')
plt.title('Histogram of Second-to-Last Column')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Histogram for the last column (after log transformation)
plt.subplot(1, 2, 2)
plt.hist(df.iloc[:, -1], bins=40, edgecolor='black')
plt.title('Histogram of Last Column (Log Transformed)')
plt.xlabel('Log-Transformed Value')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

'''