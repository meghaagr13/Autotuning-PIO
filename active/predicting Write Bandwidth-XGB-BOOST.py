#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomTreesEmbedding, ExtraTreesClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import BernoulliNB
import pickle
from sklearn.externals import joblib
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
import os 


# In[9]:


projectdir = "/home/meghaagr/project/progress/"
data = pd.read_csv('../S3DIOstats.txt', delim_whitespace=True)
data = data.sample(frac=1).reset_index(drop=True)
print(data.head(5))
data.iloc[:,5].plot()


# In[10]:


array = data.values

print(data.head())
print(data.shape)
data.columns=['col1','col2','col3','col4','col5','col6','col7','col8','col9','col10','col11','col12','col13','col14','col15','col16','col17']
z = data['col2'].str.split('-')
print(z.head())
data['col13']=data['col13'].eq("enable").mul(1)
data['col14']=data['col14'].eq("enable").mul(1)
data['col15']=data['col15'].eq("enable").mul(1)
data['col16']=data['col16'].eq("enable").mul(1)


# In[11]:


X = pd.DataFrame(z.tolist())
X = X.drop(6, 1)

#FEATURES
X['cb3']=data['col13']
X['cb4']=data['col14']
#X['cb5']=data['col15']
#X['cb6']=data['col16']
X['cb1']=data['col11']
X['cb2']=data['col12']
X['cb7']=data['col17']
X['ratio1'] = X.iloc[:,0].astype(int)/(X.iloc[:,3].astype(int))
X['ratio2'] = X.iloc[:,1].astype(int)/(X.iloc[:,4].astype(int))
X['ratio3'] = X.iloc[:,2].astype(int)/(X.iloc[:,5].astype(int))

#NORMALIZING 
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X.values)
X = pd.DataFrame(X_scaled)

print(type(X))
print(X.head())
X_total = X
#pd.options.display.max_rows=  1500


# In[12]:


Y = data['col3']
#Y = ((data['col7'])*1024)/data['col6'] + (data['col4']*1024)/data['col3']
Y=Y.values
Y = Y.astype(int)
print(Y[0:5])
import xgboost as xgb
from sklearn.model_selection import train_test_split
train_X, test_X, train_Y, test_Y = train_test_split(X_total,Y,test_size=0.2,random_state = 42) 
y = train_Y
X = train_X
print(X[0:5])
print(y.shape, X.shape)
print(test_X.shape,test_Y.shape)
print(y[0:5])


# In[13]:


xgb_model =xgb_model = xgb.XGBRegressor(objective="reg:linear",n_estimators=500,max_depth=12, random_state=42)

xgb_model.fit(X, y)
y_pred = xgb_model.predict(test_X)

mse=mean_squared_error(test_Y, y_pred)
print(np.sqrt(mse))


# In[14]:

print("Error in read  is ")

print(np.mean(np.abs((test_Y - y_pred) / test_Y)) * 100)


#print(test_Yr
from sklearn.metrics import r2_score
print(r2_score(test_Y,y_pred))


# In[15]:


fig, ax = plt.subplots()
ax.scatter(test_Y, y_pred)
ax.plot([test_Y.min(), test_Y.max()], [test_Y.min(), test_Y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.savefig("xgbs3dresult.png")
plt.show()


# In[28]:


# save the model to disk

modelfile = 'finalxgb.sav'
joblib.dump(xgb_model, open(modelfile, 'wb'))
scaler_filename = "scalerxgb.save"
joblib.dump(min_max_scaler, scaler_filename) 


# In[19]:


# ##### TESTING CODE FOR THIS MODEL


# import json
# with open(projectdir+'confex.json') as f:
#     data = json.load(f)

# romio_ds_read = data["mpi"]["romio_ds_read"] == "enable"
# romio_ds_write = data["mpi"]["romio_ds_write"] == "enable"
# romio_cb_read = data["mpi"]["romio_cb_read"]
# romio_cb_write = data["mpi"]["romio_cb_write"]
# cb_buffer_size = data["mpi"]["cb_buffer_size"]
# stripe_size = data["lfs"]["setstripe"]["size"]
# stripe_count = data["lfs"]["setstripe"]["count"]

# col_names =  ['cb5','cb6','cb7','cb8','cb8','cb10','cb11','cb12']
# cb_df  = pd.DataFrame(columns = col_names)
# cb_df.loc[len(cb_df)] = [romio_ds_read,romio_ds_write,stripe_size, stripe_count, cb_buffer_size,"50","50","50"]


# scaler_filename = "scalerxgb.save"
# scaler = joblib.load(scaler_filename)
# xgb_model = joblib.load(modelfile)

# print(test_X.head(1))


# col_names =  ['f0','1', '2','3','4','5']
# my_df = pd.DataFrame(columns=col_names)
# my_df.loc[len(my_df)] = [100,100,100,2,2,4]

# x = pd.concat([my_df,cb_df], axis=1)
# norm_df = pd.DataFrame(scaler.transform(x))
# print(norm_df)

# xgb_model.predict(norm_df)
# x
# test_X.head(1)
# trees.predict(test_X.head(1))


# In[ ]:





# In[ ]:




