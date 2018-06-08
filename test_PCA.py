#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:53:10 2018

@author: niharika-shimona
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')
import sys,os
import numpy as np
import pickle
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
# plt.ioff()
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
from sklearn.model_selection import GridSearchCV, KFold

import scipy.io as sio
from Data_Extraction import Split_class,create_dataset
from sklearn.decomposition import PCA as sklearnPCA


df_aut,df_cont = Split_class()
task  = 'Praxis.Avg.Total.Errs'
ds = '0'

x_aut,y_aut,x_cont,y_cont = create_dataset(df_aut,df_cont,task,'/home/niharika-shimona/Documents/Projects/Autism_Network/code/patient_data/')	

if ds == '0':	
 	x = x_cont
 	y= np.ravel(y_cont)
 	fold = 'cont'
 	
elif ds == '1':
 	x = x_aut
 	y= np.ravel(y_aut)
 	fold = 'aut'
 	
else:
 	x =np.concatenate((x_cont,x_aut),axis =0)
 	y = np.ravel(np.concatenate((y_cont,y_aut),axis =0))
 	fold = 'aut_cont'

#pathname = '/home/niharika-shimona/nsalab-mceh/Users/ndsouza4/Schizophrenia/'
#task = 'Schizophrenia_eig'
#cas = ''
#filename = pathname + task +'.mat'

#data = sio.loadmat(filename)
#subtx = 'x_'+ fold
#data_eg_corr = data[subtx]
#
#x = data_eg_corr
#subty = 'y_'+ fold
#y =np.ravel(data[subty])


rf_reg = RandomForestRegressor(n_estimators=1000,oob_score= True) 	


pca = sklearnPCA()
cast = 'pca' 
pipeline = Pipeline([('PCA',pca), ('rf_reg',rf_reg)])
n_comp = [5,10,15,20,25,30]
min_samples_split = np.int16(np.linspace(2,16,8))

p_grid = [{'PCA__n_components': n_comp, 'rf_reg__min_samples_split': min_samples_split}]

kf_total = KFold(n_splits=10,shuffle=False, random_state=6)
my_scorer = make_scorer(explained_variance_score)

lrgs = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv =kf_total, scoring =my_scorer, n_jobs=-1)
lrgs.fit(x,y)
param_best = lrgs.best_params_

y_train =[]
y_test = []
y_train_AF =[]
y_test_AF =[]
r2_test = []
mse_test =[]
learnt_models =[]
i = 0

for train,test in kf_total.split(x,y):
    
    model = pipeline.set_params(**param_best)
    model.fit(x[train],y[train])
    y_pred_train = model.predict(x[train])
    y_pred_test = model.predict(x[test])
    r2_test.append(r2_score(y[test],y_pred_test))
    mse_test.append(mean_squared_error(y[test],y_pred_test))

    y_train_AF = np.concatenate((y_train_AF,y_pred_train),axis =0)
    y_train = np.concatenate((y_train,y[train]),axis =0)
    y_test_AF = np.concatenate((y_test_AF,y_pred_test),axis =0)
    y_test = np.concatenate((y_test,y[test]),axis =0)
	
    learnt_models.append(model)
     
newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/Sanity_Check/Comparative_Analysis/PCA/'+fold+'/' + cast +'/' + task + '/'
if not os.path.exists(newpath):
 	os.makedirs(newpath)
os.chdir(newpath)

fig,ax =plt.subplots()
font = {'family' : 'normal',
         'weight' : 'bold',
         'size'   : 14}
matplotlib.rc('font', **font)

ax.scatter(y_train,y_train_AF,color ='red',label = 'train')
ax.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], 'k--', lw=4)
ax.plot([y_test.min()-2,y_test.max()+2], [y_test.min()-2, y_test.max()+2], 'k--', lw=4)
data_mean = np.mean(y_train)
ax.plot([y_test.min()-2,y_test.max()+2], [data_mean,data_mean], 'k--', lw=2)
plt.ylim(ymax = y_test.max()+5, ymin = y_test.min()-5)
plt.xlim(xmax =y_test.max()+5, xmin = y_test.min()-5)
# plt.ylim(ymax = 29, ymin = 5)
# plt.xlim(xmax =25, xmin = 5)
ax.scatter(y_test,y_test_AF,color ='green',label = 'test')
ax.legend(loc='best')

ax.set_xlabel('Measured',fontsize =14)
ax.set_ylabel('Predicted',fontsize =14)

ax.legend(loc='upper left')
matplotlib.rc('font', **font)	
plt.show()

figname = 'fig_test_pres.png'
fig.savefig(figname)   # save the figure to fil
plt.close(fig)

sio.savemat('Metrics_full.mat',{'r2_test':r2_test,'mse_test':mse_test,'y_pred':y_test_AF,'y_test':y_test,'y_pred_train':y_train_AF,'y_train':y_train})
pickle.dump(learnt_models, open('Models.p', 'wb'))
pickle.dump(kf_total, open('Kfold.p', 'wb')) 