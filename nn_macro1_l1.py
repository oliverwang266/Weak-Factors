import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from scipy.io import loadmat
from keras.regularizers import l1, l2
from keras.initializers import VarianceScaling
import pandas as pd
from sklearn.metrics import make_scorer
import argparse
import random
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)


Macro1 = loadmat("weak/FredMDlargeHor1.mat")
X = Macro1["X"] 
X = np.delete(X, 5, axis=1)
y = 100*Macro1["Y"].ravel()
n = len(y)
p = X.shape[1]

def create_model(hidden_units=[16], lambda_value=10, epochs=100, init_var=0.01):
    np.random.seed(an)
    random.seed(an)
    tf.random.set_seed(an)    
    def custom_initializer():
        return VarianceScaling(scale=init_var)
    model = Sequential()
    lr=.1/epochs
    # Add the first layer with the specified input_dim
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
    np.random.seed(an)
    random.seed(an)
    tf.random.set_seed(an)    
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
    
    model = KerasRegressor(build_fn=create_model, batch_size=16, verbose=0)
    nn_grid = {
        'hidden_units': [[8]],
        'lambda_value': 10**np.arange(-10.5,2+0.1,1.0),
        'epochs':[2,5,10,20,50] 
    }
    scorer = make_scorer(mean_squared_error, greater_is_better=False)

    grid = GridSearchCV(estimator=model, param_grid=nn_grid, scoring=scorer, cv=10,n_jobs=-1)
    grid_result = grid.fit(X_train, y_train-y_mean)

    # Predict on the test set
    y_predhat = grid_result.best_estimator_.predict(X_pred,verbose=0)

    f=open("weak/macro1_l1_result.txt","a")
    f.write(f"{an} {((y_pred - y_mean-y_predhat) ** 2).mean()} { ((y_pred - y_mean) ** 2).mean()} {grid_result.best_params_['epochs']} {grid_result.best_params_['lambda_value']}"+'\n')
    return 1
#an:1-45
fit_models(an)

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Assuming the file is space-separated, adjust if your file has a different format
file_path = 'macro1_l1_result.txt'  # Replace with the path to your file
df = pd.read_csv(file_path, sep=' ', header=None)  # Use the appropriate separator

# Calculate the mean of each column
means = df.mean()
print("Column Means:\n", means)
print(1-means[1]/means[2])
# Plot histograms for the last two columns
plt.figure(figsize=(12, 6))

# Histogram for the second-to-last column
plt.subplot(1, 2, 1)
plt.hist(df.iloc[:, -2], bins=1000, color='blue', edgecolor='black')
plt.title('Histogram of Second-to-Last Column')

# Histogram for the last column, after taking log10
plt.subplot(1, 2, 2)
plt.hist(np.log10(df.iloc[:, -1]), bins=40, color='green', edgecolor='black')
plt.title('Log10 Histogram of Last Column')

plt.tight_layout()
plt.show()

'''