
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeCV, LassoCV
import xgboost as xgb
from xgboost import XGBRegressor, DMatrix
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import random
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

import os
from sklearn.ensemble import RandomForestRegressor
import random 
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pickle
import xgboost as xgb

np.random.seed(1000)

import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

# pool_size = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])



ne=300



#os.chdir("data1957-2022")
filepath='weak/XGB%d_weights.json'%(an)
filepath1 = 'weak/XGB_model/XGBm_%d.json'%(an)
def myfunc(year):
    np.random.seed(year*1000)
    random.seed(year*1000)
    data = pd.read_csv("data1957-2022/data.csv",delimiter=',')
    ret = pd.read_csv("data1957-2022/ret.csv",delimiter=',').values[:,0]
    date = pd.read_csv("data1957-2022/date.csv",delimiter=',').values[:,0]
    sic2_x = pd.read_csv("data1957-2022/sic2_x.csv",delimiter=',').values  
    rn = year
    train_len=18
    test_len=12
    oos_len=1 ### 30 Periods
    t00=19570300
      
      
    t0=19570000+oos_len*rn*10000
    t1=t0+train_len*10000
    t2=t1+test_len*10000
    t3=t2+oos_len*10000
    
    ind=(date<=t1)*(date>=t00)
    xtrain=data[ind].values
    ytrain=ret[ind]
    #wtrain=weight[ind]
    #trainper=per[ind]
    traindate=date[ind]
    sic2_xtrain=sic2_x[ind,:]
      
    ind=(date<=t2)*(date>=t1)
    xtest=data[ind].values
    ytest=ret[ind]
      #wtest=weight[ind]
      #testper=per[ind]
    testdate=date[ind]
    sic2_xtest=sic2_x[ind,:]
      
    ind=(date>=t2)*(date<=t3)
    xoos=data[ind].values
    yoos=ret[ind]
      
    del data
      
      #woos=weight[ind]
      #oosper=per[ind]
    oosdate=date[ind]
    sic2_xoos=sic2_x[ind,:]
      
      # w1=np.zeros(len(traindate))
      # u=np.unique(traindate)
      # for i in range(len(u)):
      #     ind=traindate==u[i]
      #     w1[ind]=1.0/np.sum(ind)
      # w1=w1/np.sum(w1)
      # wtrain0=w1+0.0
      # #wtrain0=np.ones(wtrain)/1.0/len(wtrain)
      # wtrain=wtrain/np.sum(wtrain)
      
      # w1=np.zeros(len(testdate))
      # u=np.unique(testdate)
      # for i in range(len(u)):
      #     ind=testdate==u[i]
      #     w1[ind]=1.0/np.sum(ind)
      # w1=w1/np.sum(w1)
      # wtest0=w1+0.0
      # #wtest0=np.ones(wtest)/1.0/len(wtest)
      # wtest=wtest/np.sum(wtest)
      
      # mtrain=np.sum(wtrain0*ytrain)
      # mtest=np.sum(wtest0*ytest)
      
    mtrain=np.mean(ytrain)
    mtest=np.mean(ytest)
  ### Times All Y_t ###
    ts=pd.read_csv('data1957-2022/tspredictors_1950.csv',delimiter=',')
    d=ts['date'].values
  #  yscale=np.array([0.33,0.33,1,50,10,50,5,50])

    n1=xtrain.shape[0]
    n2=xtrain.shape[1]
    ynum=range(1,13)
    ynum=[1,2,3,4,5,6,7,8]
    xtrain=np.hstack((xtrain,np.zeros((n1,n2*len(ynum))),sic2_xtrain))
    ad=np.unique(traindate)
    for i in range(len(ad)):
      ind=traindate==ad[i]
      weizhi=(np.arange(len(d)))[d==np.floor(ad[i]/100)]-1
      for j in range(len(ynum)):
          yt=ts.iloc[weizhi,ynum[j]].values[0]#*yscale[j]
          xtrain[ind,(n2*(j+1)):(n2*(j+2))]=xtrain[ind,0:n2]*yt

    n1=xtest.shape[0]
    n2=xtest.shape[1]
    ynum=range(1,13)
    ynum=[1,2,3,4,5,6,7,8]
    xtest=np.hstack((xtest,np.zeros((n1,n2*len(ynum))),sic2_xtest))
    ad=np.unique(testdate)
    for i in range(len(ad)):
      ind=testdate==ad[i]
      weizhi=(np.arange(len(d)))[d==np.floor(ad[i]/100)]-1
      for j in range(len(ynum)):
          yt=ts.iloc[weizhi,ynum[j]].values[0]#*yscale[j]
          xtest[ind,(n2*(j+1)):(n2*(j+2))]=xtest[ind,0:n2]*yt

    n1=xoos.shape[0]
    n2=xoos.shape[1]
    ynum=range(1,13)
    ynum=[1,2,3,4,5,6,7,8]
    xoos=np.hstack((xoos,np.zeros((n1,n2*len(ynum))),sic2_xoos))
    ad=np.unique(oosdate)
    for i in range(len(ad)):
      ind=oosdate==ad[i]
      weizhi=(np.arange(len(d)))[d==np.floor(ad[i]/100)]-1
      for j in range(len(ynum)):
          yt=ts.iloc[weizhi,ynum[j]].values[0]
          xoos[ind,(n2*(j+1)):(n2*(j+2))]=xoos[ind,0:n2]*yt
    

    del ret
    del date
    del sic2_x
    del ind
    del testdate
    del traindate
    del oosdate
    del yt
    xmean=np.vstack((xtrain,xtest)).mean(axis=0)
    sd=np.vstack((xtrain,xtest)).std(axis=0)
    for i in range(xoos.shape[1]):
      s=sd[i]
      m=xmean[i]
      xtrain[:,i]=xtrain[:,i]-m
      xtest[:,i]=xtest[:,i]-m
      xoos[:,i]=xoos[:,i]-m
      if s>1e-4:
          xtrain[:,i]=xtrain[:,i]/s
          xtest[:,i]=xtest[:,i]/s
          xoos[:,i]=xoos[:,i]/s
    print('start train')
    
    xtrain=np.vstack((xtrain,xtest))
    ytrain=np.hstack((ytrain,ytest))
    del xtest
    del ytest
    
    # param_grid = {
    #     'max_depth': xgb_grid['max_depth'],
    #     'learning_rate': xgb_grid['learning_rate'],
    #     'n_estimators': xgb_grid['n_estimators']
    # }
    
    # # Create an XGBRegressor (or XGBClassifier)
    # xgb_model = XGBRegressor(objective='reg:squarederror',random_state=an, booster='gbtree', n_jobs=-1)
    
    # # Instantiate GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=xgb_model,
    #     param_grid=param_grid,
    #     n_jobs=1,
    #     cv=2,  # Number of folds in CV
    #     scoring='neg_mean_squared_error'  # Evaluation metric
    # )
    
    # # Fit the GridSearchCV
    # grid_search.fit(xtrain, ytrain)
    
    # # Retrieve the best model and parameters
    # best_model = grid_search.best_estimator_
    # # Assuming best_params is the output from GridSearchCV
    # best_params = grid_search.best_params_
    
    # # Extracting individual parameters
    # best_depth = best_params['max_depth']
    # best_learning_rate = best_params['learning_rate']
    # best_n_estimators = best_params['n_estimators']
    
    # # Converting to opt_tuning format
    # opt_tuning = [best_depth, best_learning_rate, best_n_estimators]

    
    # # Predict and evaluate (example)
    # xgb_predict= best_model.predict(xoos)
    
    # # Save your model
    # best_model.save_model(filepath1)
        
    '''    
    kf = KFold(n_splits=2)
    
    best_score = float('inf')
    opt_tuning = None
    
    for depth in xgb_grid['max_depth']:
        for lr in xgb_grid['learning_rate']:
            max_n_estimators = max(xgb_grid['n_estimators'])
            params = {
                'max_depth': depth,
                'learning_rate': lr,
                'objective': 'reg:squarederror',
                'booster': 'gbtree',
                'n_jobs': -1,
                'random_state': an
            }
    
            scores_for_estimators = {}
    
            for train_index, test_index in kf.split(xtrain):
                X_train, X_val = xtrain[train_index], xtrain[test_index]
                Y_train, Y_val = ytrain[train_index], ytrain[test_index]
    
                dtrain = xgb.DMatrix(X_train, label=Y_train)
                dval = xgb.DMatrix(X_val, label=Y_val)
    
                model = xgb.train(params, dtrain, num_boost_round=max_n_estimators)
    
                for n_estimators in xgb_grid['n_estimators']:
                    preds = model.predict(dval, iteration_range=(0, n_estimators))
                    score = mean_squared_error(Y_val, preds)
    
                    if n_estimators not in scores_for_estimators:
                        scores_for_estimators[n_estimators] = []
                    scores_for_estimators[n_estimators].append(score)
                del dtrain, dval, preds, X_train, X_val, Y_train, Y_val

            for n_estimators, scores in scores_for_estimators.items():
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    opt_tuning = [depth, lr, n_estimators]
    
    final_params = {
        'max_depth': opt_tuning[0],
        'learning_rate': opt_tuning[1],
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'n_jobs': -1,
        'random_state': an
    }
    final_model = xgb.train(final_params, xgb.DMatrix(xtrain, label=ytrain), num_boost_round=opt_tuning[2])
    # 保存模型
    final_model.save_model(filepath1)
    '''
        # 创建一个新的模型实例
    final_model = xgb.Booster()
    
    # 加载之前保存的模型
    final_model.load_model(filepath1)
        
    xoos_dmatrix = xgb.DMatrix(xoos)
    xgb_predict = final_model.predict(xoos_dmatrix)

    temp=[an, year,
        ((np.squeeze(np.asarray(yoos)) - xgb_predict) ** 2).mean(),
        np.mean(np.power(yoos,2))]
    with open("weak/finance2_xgb_result.txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f,temp,newline=" ")
    # with open("weak/finance2_xgb_result.txt", "ab") as f:
    #     f.write(b"\n")
    #     np.savetxt(f,temp+opt_tuning,newline=" ")
    return(1)

xgb_grid = {
    'learning_rate': [10**i for i in [-1,-2,-3]],
    'max_depth': [1, 2,3, 4,5],    
    'n_estimators': [10,20,50,100,200,300,400,500],
}

# an from 0-34
myfunc(an)



'''
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the file
# Replace 'sep' with the appropriate separator if the file is not a CSV
df = pd.read_csv('finance2_xgb_result.txt', sep=' ', header=None, usecols=[0, 1, 2, 3, 4,5])
column_means = df.mean()

# Print the means
print(column_means)

# Step 2: Select the last three columns
last_three_columns = df.iloc[:, -3:]

fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 创建三个并排的子图

for i, column in enumerate(last_three_columns):
    ax = axs[i]
    data = df[column]
    
    # 对倒数第二列取 log10
    if i == 1:
        data = np.log10(data[data > 0])  # 确保数据大于0再取log

    ax.hist(data, bins=20)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
'''