
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd
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
import random
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

p = X.shape[1]


def fit_models(X, y, index_train, index_pred, num2):
      np.random.seed(an)
      random.seed(an)
      y_train = y[index_train]
      X_train = X[index_train, :]
      y_pred = y[index_pred]
      X_test = X[index_pred, :]

      X_mean = X_train.mean(axis=0)
      X_std = X_train.std(axis=0)
      y_mean = y_train.mean()
      X_train = (X_train - X_mean) / X_std
      X_test = (X_test - X_mean) / X_std
      
      dtrain = xgb.DMatrix(X_train, label=y_train-y_mean)
      dtest = xgb.DMatrix(X_test, label=y_pred-y_mean)
      
      param_grid = {
          'max_depth': [1, 2, 3, 4, 5],
          'learning_rate': [0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,1e-13,1e-14,1e-15],
          'max_n_estimators': 20,
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


      f=open("weak/micro1_xgb_result.txt","a")
      f.write(f"{num2} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {best_params['max_depth']} {best_params['learning_rate']} {best_params['n_estimators']}"+'\n')

      return 1
      

num1 = 0
num2 = 0
result = []
f = open("weak/Micro1_index.txt", "r")
lines = f.readlines()

#an:1-104
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
file_path = 'micro1_xgb_result.txt'
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None)
df.columns = ['i', 'nn', 'mean']

final_temp = pd.DataFrame(np.zeros((1, 1)))
for i in range(13):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    temp[:-1] = 1 - temp[:-1]/temp[-1]
    final_temp.iloc[0] += temp[:-1].values/13
print(final_temp)


# Assuming you have already read the DataFrame
df = pd.read_csv(file_path, sep=' ', header=None)

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
axes[1].hist(np.log10(df_last_three.iloc[:, 1]), bins=40)
axes[1].set_title('Second Last Column Log Transformed Histogram')

# Histogram for the last column
axes[2].hist(df_last_three.iloc[:, 2], bins=40)
axes[2].set_title('Last Column Histogram')

# Display the histograms
plt.tight_layout()
plt.show()
'''