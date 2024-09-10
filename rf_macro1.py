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
X=np.delete(X, 5, axis=1)
y = Macro1["Y"].ravel()
n = len(y)
p = X.shape[1]



def fit_models(i):
    index_train = list(range(179+(i-1)*12))
    index_pred = list(range(179+(i-1)*12, 191+(i-1)*12))
    y_train = y[index_train]
    X_train = X[index_train, :]
    
    y_pred = y[index_pred]
    X_pred = X[index_pred, :]

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    X_train = (X_train - X_mean) / X_std
    X_pred = (X_pred - X_mean) / X_std
    
    
    rf_grid = {
        'max_depth': np.arange(5,51,5), 
        'max_features': np.arange(2,41,2), 
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rfr = RandomForestRegressor(n_estimators=500, random_state=an)
    grid = GridSearchCV(estimator=rfr,scoring=scorer, param_grid=rf_grid, cv=10,n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)

    # Predict on the test set
    y_predhat = grid_result.best_estimator_.predict(X_pred)

    f=open("weak/macro1_rf_result.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {grid_result.best_params_['max_depth']} {grid_result.best_params_['max_features']}"+'\n')
    return [
        ((y_pred -y_predhat-y_mean) ** 2).mean(),
        ((y_pred - y_mean) ** 2).mean()
    ], grid_result.best_params_

#an:1-45
fit_models(an)



'''

file_path = 'macro1_rf_result.txt'  
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