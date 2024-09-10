import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import math
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
import random
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from multiprocessing import Pool, cpu_count
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

Macro1 = loadmat("weak/FredMDlargeHor1.mat")
X = Macro1["X"]
y = Macro1["Y"]
n = len(y)
p = X.shape[1]



def fit_models(i):
    np.random.seed(i)
    random.seed(i)
    
    
    filepath='weak/Macro1data/' 
    data = loadmat(filepath+f"Macro1_{an}.mat")
    X_train = data['X_train']
    X_pred = data['X_pred']
    X_train = np.delete(X_train, 5, axis=1)
    X_pred = np.delete(X_pred, 5, axis=1)
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
        'max_depth': range(5,41,5),
        'max_features':range(3,61,3), 
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rfr = RandomForestRegressor(n_estimators=500, random_state=an)
    grid = GridSearchCV(estimator=rfr,scoring=scorer, param_grid=rf_grid, cv=10,n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)

    # Predict on the test set
    y_predhat = grid_result.best_estimator_.predict(X_pred)

    f=open("weak/macro1b_rf_result.txt","a")
    f.write(f"{an} {((y_pred - y_predb-y_predhat) ** 2).mean()} { ((y_pred - y_predb) ** 2).mean()} {grid_result.best_params_['max_depth']} {grid_result.best_params_['max_features']}"+'\n')
    return 1
#an:1-45
fit_models(an)



'''

file_path = 'macro1b_rf_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None)
column_means = df.mean()
print(column_means)
col_1 = df.iloc[:, 1]
col_2 = df.iloc[:, 2]


print(1-col_1.mean()/col_2.mean())


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is already loaded with your data
col_1 = df.iloc[:, -2]  # Second last column
col_2 = df.iloc[:, -1]  # Last column

# Creating a figure with 2 subplots
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

# Plotting histogram for second last column
axes[0].hist(col_1, bins=20, color='blue', alpha=0.7)
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
