# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 11:59:39 2024

@author: Oliver Wang
"""
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import random
import numpy as np
import pandas as pd
from scipy.io import loadmat 
from keras.regularizers import l2, l1
from keras.initializers import VarianceScaling
from sklearn.metrics import make_scorer
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

Micro1 = loadmat("weak/Abortion_data_u13.mat")
TDums = Micro1["TDums"]
Dxmurd = Micro1["Dxmurd"]
Zmurd = Micro1["Zmurd"]
Dymurd = Micro1["Dymurd"]
data = np.column_stack((np.ones(TDums.shape[0]), TDums, Dxmurd, Zmurd, Dymurd))

n = data.shape[0]
p = data.shape[1] - 1

X = data[:, 1:p]
y = data[:, p]

def create_model(hidden_units=[16], lambda_value=10, epochs=100, init_var=0.01):
    np.random.seed(an)
    random.seed(an)
    tf.random.set_seed(an)    
    def custom_initializer():
        return VarianceScaling(scale=init_var)
    model = Sequential()
    lr=.1/epochs
    # Add the first layer with the specified input_dim
    model.add(Dense(hidden_units[0], input_dim=X.shape[1], kernel_regularizer=l2(lambda_value), 
                    activation='relu',kernel_initializer=custom_initializer()))

    # If there are more hidden layers, add them sequentially
    for units in hidden_units[1:]:
        model.add(Dense(units, kernel_regularizer=l2(lambda_value), 
                        activation='relu',kernel_initializer=custom_initializer()))

    model.add(Dense(1, kernel_regularizer=l2(lambda_value), use_bias=False,
                    activation='linear',kernel_initializer=custom_initializer()))
    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=lr))
    return model

def fit_models(X, y, index_train, index_pred, num2):
      np.random.seed(an)
      random.seed(an)
      tf.random.set_seed(an)    
      y_train = y[index_train]
      X_train = X[index_train, :]
      y_pred = y[index_pred]
      X_test = X[index_pred, :]

      X_mean = X_train.mean(axis=0)
      X_std = X_train.std(axis=0)
      y_mean = y_train.mean()
      X_train = (X_train - X_mean) / X_std
      X_test = (X_test - X_mean) / X_std

      nn_grid = {
         'epochs': [1,10,100,1000,2000],
        'lambda_value': 10**np.arange(-11,-7+0.05, 0.5)
      }
      scorer = make_scorer(mean_squared_error, greater_is_better=False)
      model = KerasRegressor(build_fn=create_model, init_var=0.001, batch_size=16, verbose=0)
      grid = GridSearchCV(estimator=model, param_grid=nn_grid,scoring=scorer, cv=10, n_jobs=-1)
      grid_result = grid.fit(X_train, y_train-y_mean)
      nn = grid_result.best_estimator_
      y_predhat = nn.predict(X_test,verbose=0)

      f=open("weak/micro1_l2_result.txt","a")
      f.write(f"{an} {num2} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {grid_result.best_params_['epochs']} {grid_result.best_params_['lambda_value']}"+'\n')
      return [
          num2,
        ((y_pred -y_predhat-y_mean) ** 2).mean(),
        ((y_pred - y_mean) ** 2).mean()
      ], grid_result.best_params_

num1 = 0
num2 = 0
result = []
f = open("weak/Micro1_index.txt", "r")
lines = f.readlines()

#an 1-104
for i in range(1, len(lines)):
    lines1 = lines[i][:-1]
    line = np.array(lines1.split(' '))
    # find the index where the value is -1000
    train_endindex = np.where(line == '-1000')[0][0]
    train_index = line[3:train_endindex].astype(int) -1
    test_endindex = np.where(line == '-2000')[0][0]
    test_index = line[train_endindex+1:test_endindex].astype(int) -1
    if i==an:
        fit_models(X, y, train_index, test_index, num2)
    if(num1 == 7):
        num2+=1
        num1 = 0
    else:
        num1+=1

'''
file_path = 'micro1_l2_result.txt'
df = pd.read_csv(file_path, sep='\s+', usecols=[1, 2, 3], header=None)
df.columns = ['i', 'nn', 'mean']

final_temp = pd.DataFrame(np.zeros((1, 1)))
for i in range(13):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    temp[:-1] = 1 - temp[:-1]/temp[-1]
    final_temp.iloc[0] += temp[:-1].values/13
print(final_temp)

df = pd.read_csv(file_path, sep='\s+',  header=None)

plt.figure(figsize=(12, 6))

# Histogram for the second-to-last column
plt.subplot(1, 2, 1)
plt.hist(np.log10(df.iloc[:, -2]), bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Second-to-Last Column')

# Histogram for the last column, after taking log10
plt.subplot(1, 2, 2)
plt.hist(np.log10(df.iloc[:, -1]), bins=100, color='green', edgecolor='black')
plt.title('Log10 Histogram of Last Column')

plt.tight_layout()
plt.show()

'''
# f2 = open("weak/micro1_l2_result.txt", "a")
# df.to_csv(f2, header=False, index=False)
# f2.close()

# f = open("weak/result_nn.txt", "a")
# f.write('micro1_l2:'+str(final_temp.iloc[0,0])+'\n')
# f.close()
