#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:01:05 2018

@author: niharika-shimona
"""

import numpy as np
from time import time
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
#plt.ioff()
import sys,glob,os
from sklearn import metrics
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
import scipy.io as sio
import pandas as pd
import scipy.stats as stat
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

x_aut = []
y_aut =[]
task ='SRS.Total.Raw.Score'
x_cont = []
y_cont =[]

#Extract data
Location = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Data/motor_visual/TD_ASD_100balanced_Dx.csv'
df_score =pd.read_table(Location,sep=',', header=0)

#divide datasets according to type
mask = df_score[df_score[task] !='#NULL!']
df_aut = mask[mask['Diagnostic.Group'] == 1]
df_cont = mask[mask['Diagnostic.Group'] != 1]

#extract visio-motor feature for aut
for ID_NO,score in zip(df_aut['sub-ID'],df_aut[task]):
    
    print score
    Location = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Data/motor_visual/sub-'+`ID_NO`+'_ica_br_despiked.csv'
    df = pd.read_table(Location,sep=',', header=None) 
    y_aut.append(float(score))
    x_aut.append(np.arctanh(stat.pearsonr(df[1],df[4])[0]))
#    x_aut.append(stat.pearsonr(df[0],df[1])[0])
    
#extract visio-motor feature for cont
for ID_NO,score in zip(df_cont['sub-ID'],df_cont[task]):
    
    Location = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Data/motor_visual/sub-'+`ID_NO`+'_ica_br_despiked.csv'
    df = pd.read_table(Location,sep=',', header=None) 
    y_cont.append(float(score))
    x_cont.append(np.arctanh(stat.pearsonr(df[1],df[4])[0]))

#linear regression model
kf_total = KFold(n_splits=10,shuffle=True, random_state=2)

y_train =[]
y_test = []
y_train_AF =[]
y_test_AF =[]
r2_test = []
mse_test =[]
learnt_models =[]
i = 0

sio.savemat('visio-motor.mat',{'x_aut':x_aut,'x_cont':x_cont,'y_aut':y_aut,'y_cont':y_cont})
x_st = np.array(x_cont,dtype=np.float32).reshape(50,1)
#scaler = StandardScaler()
#scaler.fit(x_st)
x_comp = x_st
y = np.array(y_cont,dtype=np.float32).reshape(50,1)
#print stat.pearsonr(x_comp,y)


#y_test_AF = cross_val_predict(regr, x_comp, y, cv=10)
#regr.fit(x_comp,y)
#y_test_AF = regr.predict(x_comp)
#regr.coef_
#regr.intercept_

for train,test in kf_total.split(x_comp,y):
    
    regr = linear_model.LinearRegression(fit_intercept=True, normalize=False)    

    model = regr 
    model.fit(x_comp[train],y[train])

    y_pred_train = model.predict(x_comp[train])
    y_pred_train = np.asarray(y_pred_train).reshape(np.shape(y_pred_train)[0],1)
    print [y_pred_train,y[train],x_comp[train]]

    y_pred_test = model.predict(x_comp[test])
    y_pred_test = np.asarray(y_pred_test).reshape(np.shape(y_pred_test)[0],1)
    print [y_pred_test,y[test],x_comp[test]]

    r2_test.append(r2_score(y[test],y_pred_test))
    mse_test.append(mean_squared_error(y[test],y_pred_test))
         
    y_train_AF = np.concatenate((y_train_AF,np.ravel(y_pred_train)),axis =0)
    y_train = np.concatenate((y_train,np.ravel(y[train])),axis =0)
    y_test_AF = np.concatenate((y_test_AF,np.ravel(y_pred_test)),axis =0)
    y_test = np.concatenate((y_test,np.ravel(y[test])),axis =0)
	
    learnt_models.append(model)
# 

newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/Sanity_Check/motor_visual/cont/' + task + '/'
if not os.path.exists(newpath):
 	os.makedirs(newpath)
os.chdir(newpath)

fig, ax = plt.subplots(figsize=(10,6))
font = {'family' : 'normal',
         'weight' : 'bold',
         'size'   : 14}
matplotlib.rc('font', **font)
ax.scatter(y_train,y_train_AF,color='red',label = 'train')
ax.scatter(y_test, y_test_AF,color ='darkcyan',label = 'test')
#ax.scatter(np.asarray(x_cont), np.asarray(y_cont), color ='blue',label = 'test')
ax.legend(loc='upper left')
matplotlib.rc('font', **font)

ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)

data_mean = np.mean(y_train)
ax.plot([y_test.min()-2,y_test.max()+2], [data_mean,data_mean], 'k--', lw=2)

ax.set_xlabel('Measured',fontsize=14)
ax.set_ylabel('Predicted',fontsize=14)


plt.show()

sio.savemat('Metrics_full.mat',{'y_pred':y_test_AF,'y_test':y,'y_train':y_train,'y_train_pred':y_train_AF})
figname = 'fig_test_pres_intercept.png'
fig.savefig(figname)   # save the figure to fil
plt.close(fig)
   
     