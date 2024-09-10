import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.io import loadmat
import pandas as pd
from sklearn.metrics import mean_squared_error
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

X = data[:, :p]
y = data[:, p]
X = X[:, 1:]

def fit_models(X, y, index_train, index_pred,num2):
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
        'max_depth': np.concatenate((np.arange(1, 5, 1), np.arange(5, 55, 5))),
        'max_features': np.arange(1,6, 1),
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rfr = RandomForestRegressor(n_estimators=500, random_state=an)
    grid = GridSearchCV(estimator=rfr,scoring=scorer, param_grid=rf_grid, cv=10,n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)

    # Predict on the test set
    y_predhat = grid_result.best_estimator_.predict(X_pred)

    f=open("weak/micro1_rf_result.txt","a")
    f.write(f"{num2} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {grid_result.best_params_['max_depth']} {grid_result.best_params_['max_features']}"+'\n')
    return 1


#an 1-104
f = open("weak/Micro1_index.txt", "r")
lines = f.readlines()
num1=num2=0
for i in range(1, len(lines)):
    lines1 = lines[i][:-1]
    line = np.array(lines1.split(' '))
    # find the index where the value is -1000
    train_endindex = np.where(line == '-1000')[0][0]
    train_index = line[3:train_endindex].astype(int) -1
    test_endindex = np.where(line == '-2000')[0][0]
    test_index = line[train_endindex+1:test_endindex].astype(int) -1
    if an==i:
        final = fit_models(X, y, train_index, test_index, num2)
    if(num1 == 7):
        num2+=1
        num1 = 0
    else:
        num1+=1
        
    
'''
file_path = 'micro1_rf_result.txt'
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None)
df.columns = ['i', 'nn', 'mean']

final_temp = pd.DataFrame(np.zeros((1, 1)))
for i in range(13):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    temp[:-1] = 1 - temp[:-1]/temp[-1]
    final_temp.iloc[0] += temp[:-1].values/13
print(final_temp)

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(file_path, sep='\s+',  header=None)
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
axes[1].hist(col_2, bins=20, color='green', alpha=0.7)
axes[1].set_title('Histogram of Last Column')
axes[1].set_xlabel('Values')
axes[1].set_ylabel('Frequency')

# Display the plot
plt.tight_layout()
plt.show()
'''