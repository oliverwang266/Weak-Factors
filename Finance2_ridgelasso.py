import numpy as np
import math 
from scipy.stats import rankdata
import statistics as stat
import pandas as pd
import os
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import random 
import scipy.io
import multiprocessing as mproc
from sklearn.model_selection import KFold
from itertools import product
from functools import reduce
from sklearn.linear_model import RidgeCV, LassoCV
import operator
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
import os 
#import cPickle   # if you use python2, you can decomment this line and comment next line
import _pickle as cPickle
import pickle
import random 
import sys
np.random.seed(1000)
import timeit
from sklearn.preprocessing import StandardScaler
import argparse
args = argparse.ArgumentParser()
args.add_argument("Symbol", help="Symbol of A Stock")
arg = args.parse_args()
number = arg.Symbol
an=int(number)




def myfunc(year):
  np.random.seed(year*1000)
  data = pd.read_csv("data1957-2022/data.csv",delimiter=',')
  ret = pd.read_csv("data1957-2022/ret.csv",delimiter=',').values[:,0]
  date = pd.read_csv("data1957-2022/date.csv",delimiter=',').values[:,0]
  sic2_x = pd.read_csv("data1957-2022/sic2_x.csv",delimiter=',').values  
  rn = year
  train_len=18
  test_len=12
  oos_len=1 ### 30 Periods
  t00=19570300
  print("data shape:", data.shape)
  print("ret length:", len(ret))
  print("date length:", len(date))
  print("sic2_x shape:", sic2_x.shape)
        
    
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
  del xtest
  del ytest
  mtrain=np.mean(ytrain)
    
  ridge_alphas = 10 ** np.arange(6, 7+.01, .05)
  lasso_alphas = 10 ** np.arange(-3.4,-2.4+.01 ,.05)

  ridge_cv = RidgeCV(alphas=ridge_alphas, fit_intercept=False, cv=2)
  lasso_cv = LassoCV(alphas=lasso_alphas, fit_intercept=False, cv=2)
  model_dir = 'weak/Finance2_model/ridgelasso/'
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)    
  with open(os.path.join(model_dir,f'ridge_model_{year}.pkl'), 'wb') as f:
    pickle.dump(ridge_cv, f)
  with open(os.path.join(model_dir,f'lasso_model_{year}.pkl'), 'wb') as f:
    pickle.dump(lasso_cv, f)
  ridge_cv.fit(xtrain, ytrain - mtrain)
  lasso_cv.fit(xtrain, ytrain - mtrain)
  lamda_ridge = ridge_cv.alpha_
  lamda_lasso = lasso_cv.alpha_
  yhat_ridge = ridge_cv.predict(xoos) + mtrain
  yhat_lasso = lasso_cv.predict(xoos) + mtrain
  clf=LinearRegression(fit_intercept=False)
  clf.fit(xtrain,ytrain-mtrain)
  with open(os.path.join(model_dir,f'ols_model_{year}.pkl'), 'wb') as f:
    pickle.dump(clf, f)
  yhat_ols=clf.predict(xoos)+mtrain
  temp=[year,np.mean(np.power(np.squeeze(np.asarray(yoos))-yhat_ols,2)),np.mean(np.power(np.squeeze(np.asarray(yoos))-yhat_ridge,2)),np.mean(np.power(np.squeeze(np.asarray(yoos))-yhat_lasso,2)),np.mean(np.power(yoos,2)),lamda_ridge,lamda_lasso]
  with open("weak/finance2_ridgelasso.txt", "ab") as f:
      f.write(b"\n")
      np.savetxt(f,temp,newline=" ")
  return(1)

myfunc(an)


'''
file_path = 'finance2_ridgelasso.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2, 3, 4,5,6])
column_means = df.mean()
print(column_means)
# Plot histograms for the last two columns
plt.figure(figsize=(12, 6))

# Histogram for the second-to-last column
plt.subplot(1, 2, 1)
plt.hist(np.log10(df.iloc[:, -2]), bins=20, color='blue', edgecolor='black')
plt.title('Histogram of Second-to-Last Column')
plt.xlim(6, 7)

# Histogram for the last column, after taking log10
plt.subplot(1, 2, 2)
plt.hist(np.log10(df.iloc[:, -1]), bins=20, color='green', edgecolor='black')
plt.title('Log10 Histogram of Last Column')
plt.xlim(-3.55, -2.45)

plt.tight_layout()
plt.show()

'''