import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import keras
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.io import loadmat
from keras.regularizers import l1, l2
from keras.initializers import VarianceScaling
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)



Finance1 = loadmat("weak/Goyal.mat")
X = Finance1["X"]
y = 10*Finance1["Y"].flatten()

def create_model(hidden_units=[4], lambda_value=10, epochs=100, init_var=1):
    np.random.seed(an)
    random.seed(an)
    tf.random.set_seed(an)  
    lr=0.4/epochs
    def custom_initializer():
        return VarianceScaling(scale=init_var)
    model = Sequential()
    model.add(Dense(hidden_units[0], input_dim=X.shape[1], kernel_regularizer=l1(lambda_value), 
                    activation='relu',kernel_initializer=custom_initializer()))

    # If there are more hidden layers, add them sequentially
    for units in hidden_units[1:]:
        model.add(Dense(units, kernel_regularizer=l1(lambda_value), 
                        activation='relu',kernel_initializer=custom_initializer()))

    model.add(Dense(1, kernel_regularizer=l1(lambda_value), use_bias=False,
                    activation='linear',kernel_initializer=custom_initializer()))
    model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate=lr))
    return model

def fit_models(i):
    np.random.seed(i)
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
    model = KerasRegressor(build_fn=create_model, batch_size=16, verbose=0)
    nn_grid = {
        'epochs': [1,5,20],
        'lambda_value':10**np.arange(-2,1+0.1,.5),
    }
    mse_scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid = GridSearchCV(estimator=model, param_grid=nn_grid,scoring=mse_scorer, cv=10, n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)

    y_predhat = grid_result.best_estimator_.predict(X_pred,verbose=0)
    f=open("weak/finance1_l1_result.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {grid_result.best_params_['epochs']} {grid_result.best_params_['lambda_value']}"+'\n')
    return 1
np.random.seed(an)
random.seed(an)
tf.random.set_seed(an)  
#an:0-56    
output=fit_models(an)


'''
file_path = 'finance1_l1_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2, 3, 4])
column_means = df.mean()
print(column_means)
print(1-column_means[1]/column_means[2])
# Plot histograms for the last two columns
plt.figure(figsize=(12, 6))

# Histogram for the second-to-last column
plt.subplot(1, 2, 1)
plt.hist(np.log10(df.iloc[:, -2]), bins=100, color='blue', edgecolor='black')
plt.title('Histogram of Second-to-Last Column')
log_values = np.log10(df.iloc[:, -1])
log_values.replace(-np.inf, -20, inplace=True)  # Handling -inf values
# Histogram for the last column, after taking log10
plt.subplot(1, 2, 2)
plt.hist(log_values, bins=100, color='green', edgecolor='black')
plt.title('Log10 Histogram of Last Column')

plt.tight_layout()
plt.show()

'''