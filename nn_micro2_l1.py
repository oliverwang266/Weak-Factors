# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:00:35 2024

@author: Oliver Wang
"""
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer
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

Micro2 = loadmat('weak/Data1stStageEminentDomain_u80.mat')
data = Micro2['Data1stStageEminentDomain_u80']
y = data[:,-1]
X = data[:,1:-1]

f = open("weak/Micro2_index.txt", "r")
lines = f.readlines()

def create_model(hidden_units=[16], lambda_value=10, epochs=100, init_var=0.01):
    np.random.seed(an)
    random.seed(an)
    tf.random.set_seed(an)    
    lr=0.1/epochs
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

def fit_models(X, y, train_index, test_index, num2):
    np.random.seed(an)
    random.seed(an)    
    tf.random.set_seed(an)  
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_pred = y[train_index], y[test_index]
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    model = KerasRegressor(build_fn=create_model, batch_size=16, verbose=0)

    nn_grid = {
        'lambda_value': 10**np.arange(-12,-9+.2, 1.),
        'epochs': [1,10,100,1000]
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid = GridSearchCV(estimator=model, scoring=scorer,param_grid=nn_grid, cv=10, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)
    # Predict on the test set
    y_predhat = grid_result.best_estimator_.predict(X_test,verbose=0)
    f=open("weak/micro2_l1_result.txt","a")
    f.write(f"{an} {num2} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {grid_result.best_params_['epochs']} {grid_result.best_params_['lambda_value']}"+'\n')
    return [ num2,
        ((y_pred -y_predhat-y_mean) ** 2).mean(),
        ((y_pred - y_mean) ** 2).mean()
    ], grid_result.best_params_


num1 = 0
num2 = 0
results = []

#an 1-100
for i in range(1, len(lines)):
    lines1 = lines[i][:-1]
    line = np.array(lines1.split(' '))
    train_endindex = np.where(line == '-1000')[0][0]
    train_index = line[3:train_endindex].astype(int) -1
    test_endindex = np.where(line == '-2000')[0][0]
    test_index = line[train_endindex+1:test_endindex].astype(int) -1
    #fit_models(X, y, train_index, test_index, num2)
    if i==an:
        fit_models(X, y, train_index, test_index, num2)
    if(num1 == 19):
      num2+=1
      num1 = 0
    else: 
      num1+=1


'''
file_path = 'micro2_l1_result.txt'
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
plt.hist(np.log10(df.iloc[:, -2]), bins=20, edgecolor='black')
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
