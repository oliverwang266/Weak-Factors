from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.model_selection import RepeatedKFold
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


import os
from sklearn.ensemble import RandomForestRegressor
import random 
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import pickle
import random

np.random.seed(1000)

import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)

# pool_size = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])


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
    
        # Define the parameter grid
    param_grid = {
        'max_depth': lamy, # Assuming lamy is a list of values for max_depth
        'max_features': lamx, # Assuming lamx is a list of values for max_features
        'n_estimators': [500]
    }
    
    
    
    # Initialize the base model
    rf = RandomForestRegressor(n_jobs=-1,random_state=an)
    
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               cv=2, n_jobs=1, scoring='neg_mean_squared_error')
    
    # Fit the grid search to the data
    grid_search.fit(xtrain, np.ravel(ytrain))
    
    # Retrieve the best model
    best_rf = grid_search.best_estimator_
    
    # 文件夹路径
    model_dir = 'weak/rf_model'
    
    # 检查文件夹是否存在，如果不存在，则创建
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 定义文件路径
    output_file_path = os.path.join(model_dir, f'rf_model_{year}.pkl')
    
    with open(output_file_path, 'wb') as file:
        pickle.dump(best_rf, file)
    
    # If you need the best hyperparameters
    opt_tuning = [best_rf.max_depth, best_rf.max_features]
    
    # temp=100000
    # for ii in range(len(lamy)):
    #     for jj in range(len(lamx)):
    #         clf = RandomForestRegressor(max_depth =lamy[ii],n_estimators=300,max_features=lamx[jj],n_jobs=-1)
    #         clf.fit(xtrain,np.ravel(ytrain))
    #         if np.mean(np.power(np.squeeze(np.asarray(ytest))-clf.predict(xtest),2))<temp:
    #             temp=np.mean(np.power(np.squeeze(np.asarray(ytest))-clf.predict(xtest),2))
    #             s = pickle.dumps(clf)
    #             opt_tuning=[lamy[ii] , lamx[jj]]
    # clf=pickle.loads(s)

    
    temp=[year,
        ((np.squeeze(np.asarray(yoos)) - best_rf.predict(xoos)) ** 2).mean(),
        np.mean(np.power(yoos,2))]
    with open("weak/finance2_rf_result2.txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f,temp+opt_tuning,newline=" ")
    return(1)


# an from 1-34
lamx=[3,5,10,15, 20]
lamy=[1,2,3,4,5,6]
myfunc(an)
'''
with mproc.Pool(processes=pool_size) as pool:
     result_parallel = pool.map(myfunc, range(30))
'''
'''
with mproc.Pool(processes=pool_size) as pool:
     result_parallel = pool.map(myfunc, [25,27,28,29])
'''


'''
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the file
# Replace 'sep' with the appropriate separator if the file is not a CSV
df = pd.read_csv('finance2_rf_result.txt', sep=' ', header=None, usecols=[0, 1, 2, 3, 4])
column_means = df.mean()

# Print the means
print(column_means)
# Step 2: Select the last three columns
last_three_columns = df.iloc[:, -2:]

# Step 3: Plot histograms
for column in last_three_columns:
    plt.figure()
    df[column].hist(bins=40)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.show()
'''
