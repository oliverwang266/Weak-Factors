# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 14:36:20 2024

@author: Zhouyu Shen
"""
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from scipy.io import loadmat
import tensorflow as tf
from keras.regularizers import l2, l1
from keras.initializers import VarianceScaling
from keras.layers import BatchNormalization
import random
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

def create_model(hidden_units=[10], lambda_value=10,epochs=100, init_var=0.01):
    np.random.seed(an)
    random.seed(an)
    tf.random.set_seed(an)  
    lr=1/epochs
    def custom_initializer():
        return VarianceScaling(scale=init_var)
    model = Sequential()

    # Add the first layer with the specified input_dim
    model.add(BatchNormalization())
    model.add(Dense(hidden_units[0], input_dim=X.shape[1], kernel_regularizer=l2(lambda_value), 
                    activation='relu',kernel_initializer=custom_initializer()))

    # If there are more hidden layers, add them sequentially
    for units in hidden_units[1:]:
        model.add(BatchNormalization())
        model.add(Dense(units, kernel_regularizer=l2(lambda_value), 
                        activation='relu',kernel_initializer=custom_initializer()))
    model.add(BatchNormalization())
    model.add(Dense(1, kernel_regularizer=l2(lambda_value), use_bias=False,
                    activation='linear',kernel_initializer=custom_initializer()))
    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=lr))
    return model

def fit_models(X, y, index_train, index_pred):
    np.random.seed(an)
    random.seed(an)
    tf.random.set_seed(an)  
    y_train =10* y[index_train]
    X_train = X[index_train, :]
    y_pred =10* y[index_pred]
    X_test = X[index_pred, :]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    nn_grid = {
        'epochs':[50,200,800],
       # 'lambda_value': np.arange(0.5, 0.6, 0.02)
        'lambda_value': 10**np.arange(-3, 3+0.2, 0.5)
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    model = KerasRegressor(build_fn=create_model, hidden_units=[8], init_var=0.1, batch_size=16, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=nn_grid,scoring=scorer, cv=10, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)
    nn = grid_result.best_estimator_
    y_predhat = nn.predict(X_test,verbose=0)

    f=open("weak/macro2_l2_result.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()/100} { ((y_pred - y_mean) ** 2).mean()/100} {grid_result.best_params_['epochs']} {grid_result.best_params_['lambda_value']}"+'\n')
    return [
        ((y_pred -y_predhat-y_mean) ** 2).mean()/100,
        ((y_pred - y_mean) ** 2).mean()/100
    ], grid_result.best_params_


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
file_path = 'macro2_l2_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None)
column_means = df.mean()
print(column_means)
col_1 = df.iloc[:, 1] 
col_2 = df.iloc[:, 2]

result = 1 - (col_1 / col_2)

mean_value = result.mean()

print(mean_value)

# Plot histograms for the last two columns
plt.figure(figsize=(12, 6))

# Histogram for the second-to-last column
plt.subplot(1, 2, 1)
plt.hist(df.iloc[:, -2], bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Second-to-Last Column')

# Histogram for the last column, after taking log10
plt.subplot(1, 2, 2)
plt.hist(np.log10(df.iloc[:, -1]), bins=40, color='green', edgecolor='black')
plt.title('Log10 Histogram of Last Column')

plt.tight_layout()
plt.show()
'''