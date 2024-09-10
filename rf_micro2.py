# Finance1_scale.txt

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
from sklearn.metrics import make_scorer
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score

import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

Micro2 = loadmat('weak/Data1stStageEminentDomain_u80.mat')
data = Micro2['Data1stStageEminentDomain_u80']
y = data[:,-1]
X = data[:,1:-1]


def fit_models(X, y, train_index, test_index,num2):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std
    
    rf_grid = {
        'max_depth': np.arange(1, 21, 1),
        'max_features':  np.arange(1, 21, 1)
    }

    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rfr = RandomForestRegressor(n_estimators=500, random_state=an)
    grid = GridSearchCV(estimator=rfr,scoring=scorer, param_grid=rf_grid, cv=10,n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)
    y_predhat = grid_result.best_estimator_.predict(X_test)
    f=open("weak/micro2_rf_result2.txt","a")
    f.write(f"{num2} {((y_test - y_mean-y_predhat) ** 2).mean()} { ((y_test - y_mean) ** 2).mean()} {grid_result.best_params_['max_depth']} {grid_result.best_params_['max_features']}"+'\n')
    return [
        ((y_test - y_mean-y_predhat) ** 2).mean(),
        ((y_test - y_mean) ** 2).mean()
    ], grid_result.best_params_
    
f = open("weak/Micro2_index.txt", "r")
lines = f.readlines()
num1 = 0 
num2 = 0
results = []

#an: 1-100
for i in range(1, len(lines)):
    lines1 = lines[i][:-1]
    line = np.array(lines1.split(' '))
    train_endindex = np.where(line == '-1000')[0][0]
    train_index = line[3:train_endindex].astype(int) -1
    test_endindex = np.where(line == '-2000')[0][0]
    test_index = line[train_endindex+1:test_endindex].astype(int) -1
    if an==i:
        fit_models(X, y, train_index, test_index, num2)
    if(num1 == 19):
      num2+=1
      num1 = 0
    else: 
      num1+=1
      
'''
file_path = 'micro2_rf_result.txt'
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