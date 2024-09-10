# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 22:51:09 2024

@author: Oliver Wang
"""

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


feature_names = pd.read_csv("data1957-2022/data.csv", nrows=0).columns.tolist()
xgb1m_df=pd.DataFrame(columns=feature_names)

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
    ytrain=np.ravel(ytrain)
    yoos=np.ravel(yoos)  
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
                temp=[depth,lr,n_estimators,avg_score]
                if avg_score < best_score:
                    best_score = avg_score
                    opt_tuning = [depth, lr, n_estimators]
    
    params = {
        'max_depth': opt_tuning[0],
        'learning_rate':opt_tuning[1],
        'n_estimators': opt_tuning[2],
        'objective': 'reg:squarederror',
        'booster': 'gbtree',
        'random_state': an,  # 假设 an 是一个已经定义的随机状态
        'n_jobs': -1
    }
    final_model = xgb.XGBRegressor(**params)

    # 使用训练数据拟合模型
    final_model.fit(xtrain, ytrain)
    xgb_predict= final_model.predict(xoos)
    '''
    #保存模型
    filepath1 = 'weak/XGB_model/XGBmnew_%d.json'%(an)
    final_model.save_model(filepath1)
    
    final_model = xgb.Booster()
    filepath1 = 'weak/XGB_model/XGBmnew_%d.json'%(an)
    #加载之前保存的模型
    final_model.load_model(filepath1)
    xoos_dmatrix = xgb.DMatrix(xoos,label=yoos)
    xgb_predict = final_model.predict(xoos_dmatrix)
    
    r2_xgb1m=1-np.sum(np.power(xgb_predict-yoos,2))/np.sum(np.power(yoos,2))
    vip_scores_xgb1m={}
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
      yhat_xgb1m_mod = final_model.predict(xgb.DMatrix(xoos_mod))

      #计算每个模型新的R2
      r2_xgb1m_mod=1-np.sum(np.power(yhat_xgb1m_mod-yoos,2))/np.sum(np.power(yoos,2))
      #计算每个模型的R2变化
      vip_xgb1m=r2_xgb1m_mod-r2_xgb1m

      # 将每个模型的R2变化添加到字典中
      vip_scores_xgb1m[feature] = vip_xgb1m
      del xoos_mod,yhat_xgb1m_mod 
    xgb1m_df.loc[year] = vip_scores_xgb1m
    filename='weak/Finance2_VIP/xgb1mnew_df.csv'
    if os.path.exists(filename):
      existing_df=pd.read_csv(filename, index_col=0)
      existing_df.loc[year]=xgb1m_df.loc[year]
      existing_df.to_csv(filename)
    else:
       xgb1m_df.loc[[year]].to_csv(filename)

    
    '''
    combined_data=np.column_stack((oosdate,xgb_predict,yoos))
    filename1 = 'weak/Finance2_VIP/xgb_prednew.txt'
    with open(filename1, 'ab') as f:
        # 保存数据，追加到文件末尾
        np.savetxt(f, combined_data, fmt='%.16f', delimiter=" ", header="oosdate, xgb_pred, y_pred")
        
    temp=[year,
        ((np.squeeze(np.asarray(yoos)) - xgb_predict) ** 2).mean(),
        np.mean(np.power(yoos,2))]
    with open("weak/finance2_xgb_resultnew.txt", "ab") as f:
        f.write(b"\n")
        np.savetxt(f,temp+opt_tuning,newline=" ")
    # with open("weak/finance2_xgb_result.txt", "ab") as f:
    #     f.write(b"\n")
    #     np.savetxt(f,temp+opt_tuning,newline=" ")
    
    return(1)

xgb_grid = {
    'learning_rate': [0.5,0.2,0.1],
    'max_depth': [1, 2,3],    
    'n_estimators': [5,10,15,20,25,30],
}

# an from 0-34
myfunc(an)



'''
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Read the file
# Replace 'sep' with the appropriate separator if the file is not a CSV
df = pd.read_csv('finance2_xgb_resultnew.txt', sep=' ', header=None, usecols=[0, 1, 2, 3, 4,5])
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

    ax.hist(data, bins=100)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()
'''
