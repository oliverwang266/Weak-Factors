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


Finance1 = loadmat("weak/Goyal.mat")
X = Finance1["X"]
y = Finance1["Y"].flatten()


def fit_models(i):
    np.random.seed(i)
    random.seed(i)
    index_train = np.arange(0, 17 + i)
    index_pred = 17 + i
    y_train = y[index_train]
    X_train = X[index_train, :]
    X_pred = X[index_pred, :].reshape(1, -1)
    y_pred = y[index_pred] 

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    y_mean = y_train.mean()

    X_train = (X_train - X_mean) / X_std
    X_pred = (X_pred - X_mean) / X_std
    
    if np.isscalar(y_pred):
        y_pred = np.array([y_pred])
    
    dtrain = xgb.DMatrix(X_train, label=y_train-y_mean)
    dtest = xgb.DMatrix(X_pred, label=y_pred-y_mean)
    
    param_grid = {
        'max_depth': [1, 2, 3, 4, 5],
        'learning_rate': [1, 0.5, 0.2, 0.1, 0.05,0.02, 0.01],
        'max_n_estimators': 10,
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
    final_model = xgb.train(best_params, dtrain, num_boost_round=best_params['n_estimators'])
    y_predhat = final_model.predict(dtest)

    f=open("weak/finance1_xgb_result.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {best_params['max_depth']} {best_params['learning_rate']} {best_params['n_estimators']}"+'\n')

    return 1

fit_models(an)



'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Replace with the path to your text file
file_path = 'finance1_xgb_result.txt'  

# Read the file
df = pd.read_csv(file_path, sep='\s+', header=None)
# Calculate the mean of each column
column_means = df.mean()
print("Column Means:\n", column_means)

# Plot histograms for the last three columns
plt.figure(figsize=(15, 5))

# Histogram for the last column
plt.subplot(1, 3, 1)
plt.hist(df.iloc[:, -1], bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Last Column')

# Histogram for the second-to-last column (logarithmic)
plt.subplot(1, 3, 2)
plt.hist(np.log(df.iloc[:, -2]), bins=20, color='green', edgecolor='black')
plt.title('Log Histogram of Second-to-Last Column')

# Histogram for the third-to-last column
plt.subplot(1, 3, 3)
plt.hist(df.iloc[:, -3], bins=20, color='red', edgecolor='black')
plt.title('Histogram of Third-to-Last Column')

plt.tight_layout()
plt.show()

'''