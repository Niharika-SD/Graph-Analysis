#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  8 11:08:21 2018

@author: niharika-shimona
"""
import networkx as nx
import numpy as np
import scipy.io as sio
import numpy as np
from time import time
import matplotlib
import sys,glob,os
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
from Data_Extraction import Split_class,create_dataset


def Create_graph(data_corr,thresh):

    "creates a weighted graph from the correlation data"    
     #anatomical ROIs from AAL
    sz = 116
    x = [i+1 for i in xrange(sz)]

    # create a graph for AAL nodes
    G = nx.Graph()
    G.add_nodes_from(x)

    graph_eg =[]
    ind = 0
    
    #unravel the correlation matrix
    for i in xrange(sz):
        for j in range(i+1,sz):
        
            #print i,j
            if (data_corr[ind] > thresh):
                
                graph_eg.append((i+1,j+1,data_corr[ind]))
            
            ind =ind+1

    G.add_weighted_edges_from(graph_eg)
    
    return G

def Extract_Centrality_measure(G):
    
    "calculate the centrality measure feature over a graph"
    
    deg_c = nx.degree_centrality(G)    
    feat = []
    
    for key,values in deg_c.items():
        feat.append(values)
        
    feat = np.asarray(feat)
    
    return feat.T
    
#pathname = '/media/nsalab/Users/ndsouza4/Schizophrenia/'
#task = 'NCC'
#cas = 'buf'
#filename = pathname + task +'.mat'

df_aut,df_cont = Split_class()
task  = 'SRS.TotalRaw.Score'
ds = '1'

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

##data = sio.loadmat(filename)
#subtx = 'x_'+ cas
data_eg_corr = x

th =0.2
#create and store graphs from correlation data
Graph =[]
Graph.append([Create_graph(data_eg_corr[i],th) for i in xrange(data_eg_corr.shape[0])])

# feature extraction
DC = []
DC.append([Extract_Centrality_measure(Graph[0][i]) for i in xrange(data_eg_corr.shape[0])])

DC = np.reshape(DC,np.shape(DC)[1:])

x = DC
#subty = 'y_'+ cas
#y =np.ravel(data[subty])


# build regression model
my_scorer = make_scorer(explained_variance_score)
rf_reg = RandomForestRegressor(n_estimators=1000,oob_score= True)
kf_total = KFold(n_splits=10,shuffle=True, random_state=6)


depth_range = np.linspace(10,20,4)
min_samples_split = np.int16(np.linspace(2,10,5))
p_grid ={'max_depth':depth_range,'min_samples_split': min_samples_split}

lrgs = GridSearchCV(estimator=rf_reg, param_grid=p_grid, cv =kf_total, scoring =my_scorer, n_jobs=-1)
lrgs.fit(x,y)
param_best = lrgs.best_params_
#
y_train =[]
y_test = []
y_train_AF =[]
y_test_AF =[]
r2_test = []
mse_test =[]
learnt_models =[]
#i = 0

for train,test in kf_total.split(x,y):
 
    x_comp= x   
    model = rf_reg.set_params(**param_best)   

#    "Unfair selection of top 5 contributing features"
#    
#    rf_reg.fit(x[train],y[train])
#    a = rf_reg.feature_importances_
#    feat_ind_sort = [b[0] for b in sorted(enumerate(a),key=lambda i:i[1])]
#
#    n_feat =10
#
#    for i in range(np.shape(x)[0]):
#    
#         b = [x[i][j] for j in feat_ind_sort[-n_feat:]]
#         b= np.reshape(b,[1,n_feat])
#        
#         if(i==0): 
#            x_comp =  b
#         else:
#            x_comp =  np.concatenate((x_comp,b), axis=0)     
#      
#    model = LinearRegression()


    model.fit(x_comp[train],y[train])
    y_pred_train = model.predict(x_comp[train])
    y_pred_test = model.predict(x_comp[test])
    r2_test.append(r2_score(y[test],y_pred_test))
    mse_test.append(mean_squared_error(y[test],y_pred_test))

    y_train_AF = np.concatenate((y_train_AF,y_pred_train),axis =0)
    y_train = np.concatenate((y_train,y[train]),axis =0)
    y_test_AF = np.concatenate((y_test_AF,y_pred_test),axis =0)
    y_test = np.concatenate((y_test,y[test]),axis =0)
	
    learnt_models.append(model)
     

newpath = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Results/Sanity_Check/Graph_Theoretic/'+`th`+'/Degree_Centrality/' + task + '/'
if not os.path.exists(newpath):
 	os.makedirs(newpath)
os.chdir(newpath)

fig,ax =plt.subplots()
font = {'family' : 'normal',        
         'size'   : 14,
          'weight' : 'bold'}
matplotlib.rc('font', **font)

ax.scatter(y_train,y_train_AF,color ='red',label = 'train')
#ax.plot([y_test.min(),y_test.max()], [y_test.min(),y_test.max()], 'k--', lw=2)
ax.plot([y_test.min()-2,y_test.max()+2], [y_test.min()-2, y_test.max()+2], 'k--', lw=3)
data_mean = np.mean(y_train)
ax.plot([y_test.min()-2,y_test.max()+2], [data_mean,data_mean], 'k--', lw=2)
plt.ylim(ymax = y_test.max()+5, ymin = y_test.min()-5)
plt.xlim(xmax =y_test.max()+5, xmin = y_test.min()-5)
# plt.ylim(ymax = 29, ymin = 5)
# plt.xlim(xmax =25, xmin = 5)
ax.scatter(y_test,y_test_AF,color ='darkcyan',label = 'test')
ax.legend(loc='upper left')
matplotlib.rc('font', **font)

ax.set_xlabel('Measured',fontsize=14)
ax.set_ylabel('Predicted',fontsize=14)

ax.legend(loc='upper left')
	
plt.show()
figname = 'fig_test_pres.png'
fig.savefig(figname)   # save the figure to fil
plt.close(fig)

sio.savemat('Metrics_full.mat',{'r2_test':r2_test,'mse_test':mse_test,'y_pred':y_test_AF,'y_test':y_test,'y_pred_train':y_train_AF,'y_train':y_train})
pickle.dump(Graph, open('graphs.p', 'wb'))
pickle.dump(learnt_models, open('Models.p', 'wb'))
pickle.dump(kf_total, open('Kfold.p', 'wb'))

