#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 21 14:32:12 2018

@author: niharika-shimona
"""

import numpy as np
from time import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.ioff()
import sys,glob,os
from sklearn import metrics
from sklearn.metrics import mean_squared_error,explained_variance_score,mean_absolute_error,r2_score,make_scorer
import scipy.io as sio
import pandas as pd

def Split_class():
	
	"Splits the dataset into Autism and Controls"

	Location = r'/home/niharika-shimona/Documents/Projects/Autism_Network/Data/matched_data_out.xlsx'
	df = pd.read_excel(Location,0)
	mask_cont = df['Primary_Dx'] == 'None' 
	mask_aut = df['Primary_Dx'] != 'None' 

 	df_cont = df[mask_cont]
 	df_aut = df[mask_aut]
	
	return df_aut,df_cont

def evaluate_results(inner_cv,x,y,final_model,ind):

	"Performs a complete plot based evaluation of the run"    
	i =0
	sPCA_MSE =[]
	sPCA_r2=[]
	sPCA_exp=[]
	sPCA_MSE_test=[]
	sPCA_r2_test =[]
	sPCA_exp_test=[]

	for train, test in inner_cv.split(x,y):

		y_pred_train = np.asarray(final_model.predict(x[train]))
		y_pred_test = np.asarray(final_model.predict(x[test]))
		if ind > -1:
			y_pred_test =y_pred_test[:,ind]
			y_pred_train =y_pred_train[:,ind]
		
		sPCA_MSE.append(mean_squared_error(y[train], y_pred_train))
		sPCA_r2.append(r2_score(y[train], y_pred_train,multioutput='variance_weighted'))
		sPCA_exp.append(explained_variance_score(y[train], y_pred_train,multioutput='variance_weighted'))
		i= i+1
		print 'Split', i ,'\n' 
		print 'MSE : ', mean_squared_error(y[train], y_pred_train)
		print 'Explained Variance Score : ', explained_variance_score(y[train], y_pred_train,multioutput='variance_weighted')
		print 'r2 score: ' , r2_score(y[train], y_pred_train,multioutput='variance_weighted')
		fig, ax = plt.subplots()
		ax.scatter(y[train],y_pred_train)
		ax.plot([y[train].min(), y[train].max()], [y[train].min(), y[train].max()], 'k--', lw=4)
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		
		name = 'fig_'+ `i`+ '_train.png'
		fig.savefig(name)   # save the figure to fil
		plt.close(fig)

		sPCA_MSE_test.append(mean_squared_error(y[test], y_pred_test))
		sPCA_r2_test.append(r2_score(y[test], y_pred_test,multioutput='variance_weighted'))
		sPCA_exp_test.append(explained_variance_score(y[test], y_pred_test,multioutput='variance_weighted'))
		print 'MSE : ', mean_squared_error(y[test], y_pred_test)
		print 'Explained Variance Score : ', explained_variance_score(y[test], y_pred_test,multioutput='variance_weighted')
		print 'r2 score: ' , r2_score(y[test], y_pred_test,multioutput='variance_weighted')
		fig, ax = plt.subplots()
		ax.scatter(y[test],y_pred_test)
		ax.plot([y[test].min(), y[test].max()], [y[test].min(), y[test].max()], 'k--', lw=4)
		ax.set_xlabel('Measured')
		ax.set_ylabel('Predicted')
		name = `i`+ '_test.png'
		fig.savefig(name)   # save the figure to file
		plt.close(fig)

	print(np.mean(sPCA_MSE),np.mean(sPCA_r2),np.mean(sPCA_exp))
	print(np.mean(sPCA_MSE_test),np.mean(sPCA_r2_test),np.mean(sPCA_exp_test))

	return

def create_dataset(df_aut,df_cont,task,folder):
	
	"Creates the dataset according to a regression task"
	
	y_aut = np.zeros((1,1))
	y_cont = np.zeros((1,1))
	x_cont = np.zeros((1,6670))
	x_aut = np.zeros((1,6670))

	df_aut = df_aut[df_aut[task]< 7000]

	for ID_NO,score in zip(df_aut['ID'],df_aut[task]):

		filename = folder + '/Corr_' + `ID_NO` + '.mat'
		print ID_NO
		data = sio.loadmat(filename) 
		x_aut = np.concatenate((x_aut,data['corr']),axis =0)
		# x_aut = np.concatenate((x_aut,data['corr_eig_sub']),axis =0)
		y_aut = np.concatenate((y_aut,score*np.ones((1,1))),axis =0)
		
	if (task!= 'ADOS.Total'):
			
		for ID_NO,score in zip(df_cont['ID'],df_cont[task]):

			filename = folder + '/Corr_' + `ID_NO` + '.mat'
			data = sio.loadmat(filename) 
			x_cont = np.concatenate((x_cont,data['corr_eig_sub']),axis =0)
#			x_cont = np.concatenate((x_cont,data['corr']),axis =0)
			y_cont = np.concatenate((y_cont,score*np.ones((1,1))),axis =0)
			

	return x_aut[1:,:],y_aut[1:,:],x_cont[1:,:],y_cont[1:,:]

	