from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
import random
import math
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from joblib import Parallel, delayed
from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
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

# Generate a list of n_estimators values from p/10 to p, stepping by p/10

def fit_models(i):
    np.random.seed(i)
    random.seed(i)
    index_train = list(range(179+(i-1)*12))
    index_pred = list(range(179+(i-1)*12, 191+(i-1)*12))
    y_train = y[index_train]
    X_train = X[index_train, :]

    y_pred = y[index_pred]
    X_pred = X[index_pred, :]

    # Standardize the features
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()
    X_train = (X_train - X_mean) / X_std
    X_pred = (X_pred - X_mean) / X_std
    
    dtrain = xgb.DMatrix(X_train, label=y_train-y_mean)
    dtest = xgb.DMatrix(X_pred, label=y_pred-y_mean)
    
    param_grid = {
        'max_depth': [1, 2, 3, 4, 5,6],
        'learning_rate':  [0.5,0.2,0.1,0.05,0.02, 0.01,0.005],
        'max_n_estimators': 600,
        }

    
    best_score = float("inf")
    best_params = None
    max_n_estimators = param_grid['max_n_estimators']
    
    for max_depth in param_grid['max_depth']:
        for learning_rate in param_grid['learning_rate']:
            params = {
                'objective': 'reg:squarederror',
                'seed': an,
                'reg_alpha': 0,
                'reg_lambda': 0,
                'booster': 'gbtree',
                'n_jobs': -1,
                'max_depth': max_depth, 
                'learning_rate': learning_rate
            }
            

            cv_results = xgb.cv(params, dtrain, num_boost_round=max_n_estimators, nfold=10, metrics='rmse')
            
            min_rmse_index = np.argmin(cv_results['test-rmse-mean'])
            mean_rmse = cv_results['test-rmse-mean'][min_rmse_index]
            if mean_rmse < best_score:
                best_score = mean_rmse
                best_params = params.copy()
                best_params['n_estimators'] = min_rmse_index + 1  
    # 使用最佳参数训练模型
    final_model = xgb.train(best_params, dtrain, num_boost_round=best_params['n_estimators'])

    y_predhat = final_model.predict(dtest)


    f=open("weak/macro1_xgb_result.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {best_params['max_depth']} {best_params['learning_rate']} {best_params['n_estimators']}"+'\n')

    # Return the MSE
    return 1

#an:1-45
fit_models(an)

'''
file_path = 'macro1_xgb_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None)
column_means = df.mean()
print(column_means)
col_1 = df.iloc[:, 1]
col_2 = df.iloc[:, 2]


print(1-col_1.mean()/col_2.mean())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming you have already read the DataFrame
# df = pd.read_csv(file_path, sep=' ', header=None)

# Select the last three columns
df_last_three = df.iloc[:, -3:]

# Create a figure with 3 subplots
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# Histogram for the third last column
axes[0].hist(df_last_three.iloc[:, 0], bins=20)
axes[0].set_title('Third Last Column Histogram')

# Histogram for the second last column with log transformation
# Replace zeros with a very small number to avoid log(0)
df_last_three.iloc[:, 1].replace(0, np.finfo(float).eps, inplace=True)
axes[1].hist(np.log10(df_last_three.iloc[:, 1]), bins=20)
axes[1].set_title('Second Last Column Log Transformed Histogram')

# Histogram for the last column
axes[2].hist(df_last_three.iloc[:, 2], bins=40)
axes[2].set_title('Last Column Histogram')

# Display the histograms
plt.tight_layout()
plt.show()

'''