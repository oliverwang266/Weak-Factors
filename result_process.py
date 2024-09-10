import os
os.chdir('C:\\Users\Zhouyu Shen\Dropbox\PC\Desktop/research/weak signal/Can machines learn weak signals/code/nn_empirical')

import pandas as pd
import numpy as np
#Finance1
dataname='finance1'
file_path = dataname+'_l1_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
NN1=1-column_means[1]/column_means[2]
file_path = dataname+'_l2_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
NN2=1-column_means[1]/column_means[2]
file_path = dataname+'_rf_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
rf=1-column_means[1]/column_means[2]
file_path = dataname+'_xgb_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
xgb=1-column_means[1]/column_means[2]
file_path = dataname+'_scale.txt'  
df = pd.read_csv(file_path, sep=' ')
column_means = df.mean()
ols=1-column_means[0]/column_means[3]
ridge=1-column_means[1]/column_means[3]
lasso=1-column_means[2]/column_means[3]
print(dataname+':'+f"{ridge*100:.2f} {lasso*100:.2f} {ols*100:.2f} {rf*100:.2f} {xgb*100:.2f} {NN2*100:.2f} {NN1*100:.2f}")


#Finance2
dataname='finance2'
df1 = pd.read_csv("Finance2_l1_pred.txt", delimiter=" ")
df1.rename(columns = {"#":"date", "oosdate,":"nn_pred", "nn_pred,": "y_pred"}, inplace= True)
df1 = df1.iloc[:,:-1]
for col in df1.columns:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
NN1=1-np.sum( (df1.iloc[:,1]-df1.iloc[:,2])**2 )/np.sum(df1.iloc[:,2]**2)

df1 = pd.read_csv("Finance2_l2_pred.txt", delimiter=" ")
df1.rename(columns = {"#":"date", "oosdate,":"nn_pred", "nn_pred,": "y_pred"}, inplace= True)
df1 = df1.iloc[:,:-1]
for col in df1.columns:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
NN2=1-np.sum( (df1.iloc[:,1]-df1.iloc[:,2])**2 )/np.sum(df1.iloc[:,2]**2)

df1 = pd.read_csv("Finance2_rf_pred.txt", delimiter=" ")
df1.rename(columns = {"#":"date", "oosdate,":"nn_pred", "nn_pred,": "y_pred"}, inplace= True)
df1 = df1.iloc[:,:-1]
for col in df1.columns:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
rf=1-np.sum( (df1.iloc[:,1]-df1.iloc[:,2])**2 )/np.sum(df1.iloc[:,2]**2)

df1 = pd.read_csv("Finance2_xgb_pred.txt", delimiter=" ")
df1.rename(columns = {"#":"date", "oosdate,":"nn_pred", "nn_pred,": "y_pred"}, inplace= True)
df1 = df1.iloc[:,:-1]
for col in df1.columns:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
xgb=1-np.sum( (df1.iloc[:,1]-df1.iloc[:,2])**2 )/np.sum(df1.iloc[:,2]**2)
df1 = pd.read_csv("Finance2_ridgelasso_pred.txt", delimiter=" ")
df1.rename(columns = {"oosdate,":"date"}, inplace= True)
df1 = df1.iloc[:,:-1]
df1.rename(columns={
    '#': 'date', 
    'date': 'ridge', 
    'ridge_pred,': 'lasso', 
    'lasso_pred,': 'ols', 
    'ols_pred,': 'y_pred'
}, inplace=True)
for col in df1.columns:
    df1[col] = pd.to_numeric(df1[col], errors='coerce')
ols=1-np.sum( (df1['ols']-df1['y_pred'])**2 )/np.sum(df1['y_pred']**2)
ridge=1-np.sum( (df1['ridge']-df1['y_pred'])**2 )/np.sum(df1['y_pred']**2)
lasso=1-np.sum( (df1['lasso']-df1['y_pred'])**2 )/np.sum(df1['y_pred']**2)
print(dataname+':'+f"{ridge*100:.2f} {lasso*100:.2f} {ols*100:.2f} {rf*100:.2f} {xgb*100:.2f} {NN2*100:.2f} {NN1*100:.2f}")


#Macro1
dataname='Macro1'
file_path = dataname+'_l1_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
NN1=1-column_means[1]/column_means[2]
file_path = dataname+'_l2_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
NN2=1-column_means[1]/column_means[2]
file_path = dataname+'_rf_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
rf=1-column_means[1]/column_means[2]
file_path = dataname+'_xgb_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
xgb=1-column_means[1]/column_means[2]
file_path = dataname+'_scale.txt'  
df = pd.read_csv(file_path, sep=' ')
column_means = df.mean()
ols=1-column_means[0]/column_means[3]
ridge=1-column_means[1]/column_means[3]
lasso=1-column_means[2]/column_means[3]
print(dataname+':'+f"{ridge*100:.2f} {lasso*100:.2f} {ols*100:.2f} {rf*100:.2f} {xgb*100:.2f} {NN2*100:.2f} {NN1*100:.2f}")


dataname='Macro1b'
file_path = dataname+'_l1_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
NN1=1-column_means[1]/column_means[2]
file_path = dataname+'_l2_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
NN2=1-column_means[1]/column_means[2]
file_path = dataname+'_rf_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
rf=1-column_means[1]/column_means[2]
file_path = dataname+'_xgb_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
column_means = df.mean()
xgb=1-column_means[1]/column_means[2]
file_path = dataname+'_scale.txt'  
df = pd.read_csv(file_path, sep=' ')
column_means = df.mean()
ols=1-column_means[0]/column_means[3]
ridge=1-column_means[1]/column_means[3]
lasso=1-column_means[2]/column_means[3]
print(dataname+':'+f"{ridge*100:.2f} {lasso*100:.2f} {ols*100:.2f} {rf*100:.2f} {xgb*100:.2f} {NN2*100:.2f} {NN1*100:.2f}")


#Macro2
dataname='Macro2'
file_path = dataname+'_l1_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
NN1=np.mean(1-df.iloc[:,1]/df.iloc[:,2])
NN1std=np.std(1-df.iloc[:,1]/df.iloc[:,2])
file_path = dataname+'_l2_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
NN2=np.mean(1-df.iloc[:,1]/df.iloc[:,2])
NN2std=np.std(1-df.iloc[:,1]/df.iloc[:,2])
file_path = dataname+'_rf_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
rf=np.mean(1-df.iloc[:,1]/df.iloc[:,2])
rfstd=np.std(1-df.iloc[:,1]/df.iloc[:,2])
file_path = dataname+'_xgb_result.txt'  
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
df = pd.read_csv(file_path, sep=' ', header=None, usecols=[0, 1, 2])
xgb=np.mean(1-df.iloc[:,1]/df.iloc[:,2])
xgbstd=np.std(1-df.iloc[:,1]/df.iloc[:,2])
file_path = dataname+'_scale.txt'  
df = pd.read_csv(file_path, sep=' ')
ols=np.mean(1-df.iloc[:,0]/df.iloc[:,3])
olsstd=np.std(1-df.iloc[:,0]/df.iloc[:,3])
ridge=np.mean(1-df.iloc[:,1]/df.iloc[:,3])
ridgestd=np.std(1-df.iloc[:,1]/df.iloc[:,3])
lasso=np.mean(1-df.iloc[:,2]/df.iloc[:,3])
lassostd=np.std(1-df.iloc[:,2]/df.iloc[:,3])
print(dataname+':'+f"{ridge*100:.2f} {lasso*100:.2f} {ols*100:.2f} {rf*100:.2f} {xgb*100:.2f} {NN2*100:.2f} {NN1*100:.2f}")
print(dataname+'std:'+f"{ridgestd*100:.2f} {lassostd*100:.2f} {olsstd*100:.2f} {rfstd*100:.2f} {xgbstd*100:.2f} {NN2std*100:.2f} {NN1std*100:.2f}")


#Micro1 
dataname='Micro1'
file_path = dataname+'_l1_result.txt'  
df = pd.read_csv(file_path, sep='\s+', usecols=[1, 2, 3], header=None)
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(13):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
NN1=np.mean(final)
NN1std=np.std(final)
file_path = dataname+'_l2_result.txt'  
df = pd.read_csv(file_path, sep='\s+', usecols=[1, 2, 3], header=None)
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(13):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
NN2=np.mean(final)
NN2std=np.std(final)
file_path = dataname+'_rf_result.txt' 
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None) 
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(13):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
rf=np.mean(final)
rfstd=np.std(final)
file_path = dataname+'_xgb_result.txt' 
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None) 
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(13):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
xgb=np.mean(final)
xgbstd=np.std(final)
file_path = dataname+'_scale.txt'  
df = pd.read_csv(file_path, sep=' ')
df.columns = ['i', 'ols','ridge','lasso', 'mean']
final=[]
for i in range(1,14):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['ols']/temp['mean'])
ols=np.mean(final)
olsstd=np.std(final)
final=[]
for i in range(1,14):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['ridge']/temp['mean'])
ridge=np.mean(final)
ridgestd=np.std(final)
final=[]
for i in range(1,14):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['lasso']/temp['mean'])
lasso=np.mean(final)
lassostd=np.std(final)
print(dataname+':'+f"{ridge*100:.2f} {lasso*100:.2f} {ols*100:.2f} {rf*100:.2f} {xgb*100:.2f} {NN2*100:.2f} {NN1*100:.2f}")
print(dataname+'std:'+f"{ridgestd*100:.2f} {lassostd*100:.2f} {olsstd*100:.2f} {rfstd*100:.2f} {xgbstd*100:.2f} {NN2std*100:.2f} {NN1std*100:.2f}")


###micro2
dataname='Micro2'
file_path = dataname+'_l1_result.txt'  
df = pd.read_csv(file_path, sep='\s+', usecols=[1, 2, 3], header=None)
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
NN1=np.mean(final)
NN1std=np.std(final)
file_path = dataname+'_l2_result.txt'  
df = pd.read_csv(file_path, sep='\s+', usecols=[1, 2, 3], header=None)
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
NN2=np.mean(final)
NN2std=np.std(final)
file_path = dataname+'_rf_result.txt' 
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None) 
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
rf=np.mean(final)
rfstd=np.std(final)
file_path = dataname+'_xgb_result.txt' 
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None) 
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
xgb=np.mean(final)
xgbstd=np.std(final)
file_path = dataname+'_scale.txt'  
df = pd.read_csv(file_path, sep=' ')
df.columns = ['i', 'ols','ridge','lasso', 'mean']
final=[]
for i in range(1,6):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['ols']/temp['mean'])
ols=np.mean(final)
olsstd=np.std(final)
final=[]
for i in range(1,6):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['ridge']/temp['mean'])
ridge=np.mean(final)
ridgestd=np.std(final)
final=[]
for i in range(1,6):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['lasso']/temp['mean'])
lasso=np.mean(final)
lassostd=np.std(final)
print(dataname+':'+f"{ridge*100:.2f} {lasso*100:.2f} {ols*100:.2f} {rf*100:.2f} {xgb*100:.2f} {NN2*100:.2f} {NN1*100:.2f}")
print(dataname+'std:'+f"{ridgestd*100:.2f} {lassostd*100:.2f} {olsstd*100:.2f} {rfstd*100:.2f} {xgbstd*100:.2f} {NN2std*100:.2f} {NN1std*100:.2f}")


dataname='Micro2b'
file_path = dataname+'_l1_result.txt'  
df = pd.read_csv(file_path, sep='\s+', usecols=[1, 2, 3], header=None)
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
NN1=np.mean(final)
NN1std=np.std(final)
file_path = dataname+'_l2_result.txt'  
df = pd.read_csv(file_path, sep='\s+', usecols=[1, 2, 3], header=None)
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
NN2=np.mean(final)
NN2std=np.std(final)
file_path = dataname+'_rf_result.txt' 
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None) 
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
rf=np.mean(final)
rfstd=np.std(final)
file_path = dataname+'_xgb_result.txt' 
df = pd.read_csv(file_path, sep='\s+', usecols=[0, 1, 2], header=None) 
df.columns = ['i', 'nn', 'mean']
final=[]
for i in range(5):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['nn']/temp['mean'])
xgb=np.mean(final)
xgbstd=np.std(final)
file_path = dataname+'_scale.txt'  
df = pd.read_csv(file_path, sep=' ')
df.columns = ['i', 'ols','ridge','lasso', 'mean']
final=[]
for i in range(1,6):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['ols']/temp['mean'])
ols=np.mean(final)
olsstd=np.std(final)
final=[]
for i in range(1,6):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['ridge']/temp['mean'])
ridge=np.mean(final)
ridgestd=np.std(final)
final=[]
for i in range(1,6):  
    temp = df[df.iloc[:,0] == i].mean()[1:]
    final.append(1 - temp['lasso']/temp['mean'])
lasso=np.mean(final)
lassostd=np.std(final)
print(dataname+':'+f"{ridge*100:.2f} {lasso*100:.2f} {ols*100:.2f} {rf*100:.2f} {xgb*100:.2f} {NN2*100:.2f} {NN1*100:.2f}")
print(dataname+'std:'+f"{ridgestd*100:.2f} {lassostd*100:.2f} {olsstd*100:.2f} {rfstd*100:.2f} {xgbstd*100:.2f} {NN2std*100:.2f} {NN1std*100:.2f}")


