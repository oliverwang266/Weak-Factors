from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import pandas as pd 
import math
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV

from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import pinv
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

Finance1 = loadmat("weak/Goyal.mat")
X = Finance1["X"]
y = Finance1["Y"].flatten()

def fit_models(i):
    np.random.seed(1000+i)
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
    
    rf_grid = {
        'max_depth': np.arange(1, 11, 1),
        'max_features': [1,2,3,4,5,6,7,8,9,10]
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    rfr = RandomForestRegressor(n_estimators=500, random_state=an)
    grid = GridSearchCV(estimator=rfr,scoring=scorer, param_grid=rf_grid, cv=10,n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)

    y_predhat = grid_result.best_estimator_.predict(X_pred)
    f=open("weak/finance1_rf_result2.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {grid_result.best_params_['max_depth']} {grid_result.best_params_['max_features']}"+'\n')
    return 1
#0-56    
output=fit_models(an)

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming the file is space-separated, adjust if your file has a different format
file_path = "finance1_rf_result2.txt"
df = pd.read_csv(file_path, sep=' ', header=None)

# Calculate the mean of each column
means = df.mean()

print("Column Means:")
for col, mean in means.iteritems():
    print(f"{col}: {mean:.8f}")
print(1-means[1]/means[2])
# Plot histograms for the last two columns
plt.figure(figsize=(12, 6))

# Histogram for the second-to-last column
plt.subplot(1, 2, 1)
plt.hist(df.iloc[:, -2], bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Second-to-Last Column')

# Histogram for the last column, after taking log10
plt.subplot(1, 2, 2)
plt.hist((df.iloc[:, -1]), bins=100, color='green', edgecolor='black')
plt.title('Histogram of Last Column')

plt.tight_layout()
plt.show()
'''