# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 17:12:56 2024

@author: Zhouyu Shen
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 00:07:04 2024

@author: Zhouyu Shen
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 23:29:26 2023

@author: Zhouyu Shen
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 19:47:08 2023

@author: Zhouyu Shen
"""
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import tensorflow as tf
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.initializers import VarianceScaling
import pandas as pd 
from keras.optimizers import Adam, SGD
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import RidgeCV, LassoCV
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from joblib import dump,load
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
import xgboost as xgb
import random
np.random.seed(1000)

import argparse
from keras.callbacks import EarlyStopping
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.layers import ReLU
from keras.initializers import RandomUniform
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('integer', metavar='N', type=int, help='an integer for the accumulator')
arg = parser.parse_args()
an=int(arg.integer)


# pool_size = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])

def create_model(lbd, epochs):
    lr=1/epochs
    model = Sequential()
    model.add(Dense(32, 
                    activation='linear', 
                    input_shape=(920,), 
                    kernel_regularizer=regularizers.l2(lbd),
                    kernel_initializer=VarianceScaling(scale=0.01)))
    #model.add(BatchNormalization())
    model.add(ReLU(negative_slope=0.0))

    # model.add(Dense(16, 
    #                 activation='linear',
    #                 kernel_regularizer=regularizers.l2(lbd),
    #                 kernel_initializer=VarianceScaling(scale=0.01)))
    # #model.add(BatchNormalization())
    # model.add(ReLU(negative_slope=0.0))

    # model.add(Dense(8,
    #                 activation='linear',
    #                 kernel_regularizer=regularizers.l2(lbd),
    #                 kernel_initializer=VarianceScaling(scale=0.01)))
    # #model.add(BatchNormalization())
    # model.add(ReLU(negative_slope=0.0))

    model.add(Dense(1,use_bias=False,
                    kernel_regularizer=regularizers.l2(lbd),
                    kernel_initializer=VarianceScaling(scale=0.01)))

    optimizer = SGD(learning_rate=lr)
    model.compile(loss="mean_squared_error", optimizer=optimizer, metrics=['mse'])
    
    return model 


resultpath = f"weak/finance2_l2_result3.txt"
feature_names = pd.read_csv("data1957-2022/data.csv", nrows=0).columns.tolist()
NN2m_df=pd.DataFrame(columns=feature_names)
def myfunc(year):
    np.random.seed(year*1000)
    random.seed(year*1000)
    tf.random.set_seed(year*1000)    
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

    xtrain=np.vstack((xtrain,xtest))
    ytrain=np.hstack((ytrain,ytest))
    
    del xtest,ytest
    
    ymean=ytrain.mean()
    model = KerasRegressor(build_fn=create_model, batch_size=10000, verbose=0)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    grid = GridSearchCV(estimator=model, param_grid=nn_grid, scoring=scorer,cv=2,n_jobs=1)
    grid_result = grid.fit(xtrain, ytrain-ymean)
    final_model=grid_result.best_estimator_
    nn_pred = final_model.predict(xoos,verbose=0)+ymean
    
    del xtrain, ytrain
    r2_NN2m=1-np.sum(np.power(nn_pred-yoos,2))/np.sum(np.power(yoos,2))
    vip_scores_NN2m={}
    for i, feature in enumerate(feature_names):
      # 创建xtrain和xoos的副本
      xoos_mod = xoos.copy()

      # 找到特征所在的索引
      feature_index = feature_names.index(feature)
      xoos_mod[:, feature_index] = 0
      for j in range(1, 9):
          interaction_index = n2 * j + feature_index
          xoos_mod[:, interaction_index] = 0

      # 对每个模型进行预测并计算MSE
      yhat_nn2m_mod = final_model.predict(xoos_mod) +ymean

      #计算每个模型新的R2
      r2_nn2m_mod=1-np.sum(np.power(yhat_nn2m_mod-yoos,2))/np.sum(np.power(yoos,2))
      #计算每个模型的R2变化
      vip_nn2m=r2_nn2m_mod-r2_NN2m

      # 将每个模型的R2变化添加到字典中
      vip_scores_NN2m[feature] = vip_nn2m
      del xoos_mod,yhat_nn2m_mod 
    NN2m_df.loc[year] = vip_scores_NN2m
    filename='weak/Finance2_VIP/NN2m_df.csv'
    if os.path.exists(filename):
      existing_df=pd.read_csv(filename, index_col=0)
      existing_df.loc[year]=NN2m_df.loc[year]
      existing_df.to_csv(filename)
    else:
       NN2m_df.loc[[year]].to_csv(filename)
    
    opt_tuning=[grid_result.best_params_['epochs'],grid_result.best_params_['lbd']]  
    temp=[year,
        ((np.squeeze(np.asarray(yoos)) - nn_pred) ** 2).mean(),
        np.mean(np.power(yoos,2))]
    opt_tuning = np.array(opt_tuning).reshape(1, -1)
    temp = np.array(temp).reshape(1, -1)
    combined_arr = np.hstack([temp, opt_tuning])
    with open(resultpath, "ab") as f:
        f.write(b"\n")
        np.savetxt(f, combined_arr, fmt='%.16f', newline=" ")
    return(1)
    

nn_grid = {
    'epochs':[2,10,15],
    'lbd':  [10**x for x in [-4,-2,0]] 
}

# an from 0-34

myfunc(an)

'''
file_path = 'finance2_l2_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2, 3, 4])
column_means = df.mean()
print(column_means)
print(1-column_means[1]/column_means[2])
# Plot histograms for the last two columns
plt.figure(figsize=(12, 6))

# Histogram for the second-to-last column
plt.subplot(1, 2, 1)
plt.hist(df.iloc[:, -2], bins=20, color='blue', edgecolor='black')
plt.title('Epochs')

# Histogram for the last column, after taking log10
plt.subplot(1, 2, 2)
plt.hist(np.log10(df.iloc[:, -1]), bins=20, color='green', edgecolor='black')
plt.title('Log Lambda')

plt.tight_layout()
plt.show()

'''

