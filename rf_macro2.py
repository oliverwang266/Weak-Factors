# Macro2_scale.txt

from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd 
import math
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# use parallel processing to calculate the mse and r2
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV

from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import random
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
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

def fit_models(X, y, index_train, index_pred):
    np.random.seed(an)
    random.seed(an)
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
        'max_depth': range(1,21,1),
        'max_features': range(2,61,2),
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rfr = RandomForestRegressor(n_estimators=500, random_state=an)
    grid = GridSearchCV(estimator=rfr,scoring=scorer, param_grid=rf_grid, cv=10,n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)

    # Predict on the test set
    y_predhat = grid_result.best_estimator_.predict(X_pred)

    f=open("weak/macro2_rf_result.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {grid_result.best_params_['max_depth']} {grid_result.best_params_['max_features']}"+'\n')
    return [
        ((y_pred -y_predhat-y_mean) ** 2).mean(),
        ((y_pred - y_mean) ** 2).mean()
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
fit_models(X, y, train_index, test_index)





'''
file_path = 'macro2_rf_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None)
column_means = df.mean()
print(column_means)
col_1 = df.iloc[:, 1]
col_2 = df.iloc[:, 2]

result = 1 - (col_1 / col_2)

mean_value = result.mean()

print(mean_value)


import pandas as pd
import matplotlib.pyplot as plt

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