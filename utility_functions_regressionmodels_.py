#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from random import randint
import random
#Figure design
import matplotlib.pyplot as plt
params = {'figure.figsize': (8, 8),
          'figure.dpi':300,    
          'figure.titlesize':12,
          #'font.weight':'bold',
          'axes.labelsize': 20,
          'axes.titlesize':16,
          'axes.labelpad':1,
          'xtick.labelsize':30,
          'ytick.labelsize':30,
          'legend.fontsize': 12,
          'legend.title_fontsize':10,
          'lines.linewidth': 5}
plt.rcParams.update(plt.rcParamsDefault) #get back to default parameters
plt.rcParams.update(params)
import matplotlib.style as style 
style.use('bmh')


rootdir_for_datafiles = '/home/tiasha/Desktop/TiashaHD/data/Data_SubspaceProject/'


def shuffle_trials (arr): # Start from the last element and swap one by one.
    random.seed(24)
    n=len(arr)
    for i in range(n-1,0,-1):
        j = randint(0,i-1)
        arr[i],arr[j] = arr[j],arr[i] 
    return arr


def create_train_val_test_indices(no_reps,no_folds): #this method assumes you have same no of trials for test and validation. And no restriction on train trials. 
    rep_ind=shuffle_trials(np.arange(no_reps))
    no_test=int(no_reps/no_folds); 
    no_val=no_test
    no_train= no_reps-(no_val+no_test)
    train_ind=np.zeros((no_folds,no_train),dtype=int)
    val_ind=np.zeros((no_folds,no_val),dtype=int)
    test_ind=np.zeros((no_folds,no_test),dtype=int)
    for f in range(no_folds):
        train_ind[f,:]=rep_ind[:no_train]
        val_ind[f,:]=rep_ind[no_train:no_train+no_val]
        test_ind[f,:]=rep_ind[no_train+no_val:]
        rep_ind=np.roll(rep_ind, -no_test)
    return train_ind,val_ind,test_ind


def split_data2XY(data,perm):#Assumes that data is (no of voxels * no condns * no reps in a fold)
    no_vox,no_cond,no_rep=data.shape
    X=np.zeros((1, no_vox))
    Y=np.zeros((1, no_vox))
    for c in range(no_cond):
        data_percond=data[:,c,:]
        data1_percond=data_percond.T
        data2_percond=data_percond[:,perm].T
        X= np.append(X,data1_percond,axis=0)
        Y= np.append(Y,data2_percond,axis=0)
    X_test=X[1:,:]
    Y_test=Y[1:,:]
    return X_test,Y_test 


def create_data_shuffled_crossvalidated(data, train_ind,val_ind,test_ind): #Assumes that data is (no of voxels * no condns * no reps in a fold)
    data_train=data[:,:,train_ind]
    X_train,Y_train=split_data2XY(data_train,perm=[2,0,1,3])
    data_val=data[:,:,val_ind]
    X_val,Y_val=split_data2XY(data_val,perm=[1,0])
    data_test=data[:,:,test_ind]
    X_test,Y_test=split_data2XY(data_test,perm=[1,0])
    return X_train,X_val,X_test,Y_train,Y_val,Y_test


def scale_cv_data(train_data,valid_data,test_data):
    scaler_X = StandardScaler()
    train_scaled = scaler_X.fit_transform(train_data)
    valid_scaled = scaler_X.transform(valid_data)
    test_scaled = scaler_X.transform(test_data)
    return train_scaled,valid_scaled,test_scaled


def find_rank(pred_curve):
    no_ranks=pred_curve.shape[0]
    chosen_rank=0
    best=-10
    max_pred=np.max(pred_curve)
    print(max_pred,0.99*max_pred)# NSD Imagery was 0.99
    for p in range(no_ranks):
        if pred_curve[p]>best: #if better than best so far
            chosen=p #record this rank as the best
            best=pred_curve[p]
            if best>=0.99*max_pred:
                break
    return chosen


def simpleregression(X_train,X_val,X_test,y_train,y_val,y_test): 
    n_vox=X_train.shape[1]
    X_train1,X_val1,X_test1= scale_cv_data(X_train,X_val,X_test)
    y_train1,y_val1,y_test1= scale_cv_data(y_train,y_val,y_test)
    y_pred=np.zeros((y_test1.shape[0],n_vox))
    for vox in np.arange(n_vox):
        X_train_vox=X_train1[:,vox].reshape(-1, 1)
        X_test_vox=X_test1[:,vox].reshape(-1, 1)
        y_train_vox=y_train1[:,vox].reshape(-1, 1)
        y_test_vox=y_test1[:,vox].reshape(-1, 1)
        simplereg = sklearn.linear_model.LinearRegression(fit_intercept=False).fit(X_train_vox, y_train_vox)
        y_pred[:,vox] =simplereg.predict(X_test_vox).T
    return y_test1,y_pred


def multipleregression(X_train,X_val,X_test,y_train,y_val,y_test,filename_marker):
    n_vox_X=X_train.shape[1]
    n_vox_Y=y_train.shape[1]    
    ridge_coeffs= np.logspace(-3, 5, 100)
    X_train1,X_val1,X_test1= scale_cv_data(X_train,X_val,X_test)
    y_train1,y_val1,y_test1= scale_cv_data(y_train,y_val,y_test)
    beta_ridge=np.zeros([n_vox_X,n_vox_Y])
    y_ridge=np.zeros([y_train1.shape[0],n_vox_Y])
    y_pred=np.zeros((y_test1.shape[0],n_vox_Y))
    alpha_pervox=[]
    for vox in np.arange(n_vox_Y):
        y_train_vox=y_train1[:,vox].reshape(-1, 1)
        y_val_vox=y_val1[:,vox].reshape(-1, 1)
        y_test_vox=y_test1[:,vox].reshape(-1, 1)
        #ridge parameter tuning using validation set
        f_norm_error=np.zeros([len(ridge_coeffs)])
        count=0
        for alpha in ridge_coeffs:  
            ridgereg = sklearn.linear_model.Ridge(alpha=alpha, fit_intercept=False, solver='auto').fit(X_train1, y_train_vox)
            y_pred_val =ridgereg.predict(X_val1)
            f_norm_error[count] = np.linalg.norm(y_val_vox-y_pred_val, ord='fro', axis=None)
            count=count+1
        alpha_y_vox=ridge_coeffs[np.argmin(f_norm_error)]
        ridge_y_vox = sklearn.linear_model.Ridge(alpha=alpha_y_vox, fit_intercept=False, solver='auto').fit(X_train1, y_train_vox)
        #prediction on test set
        y_pred[:,vox]=ridge_y_vox.predict(X_test1).T
        #save for multivariate
        alpha_pervox=np.append(alpha_pervox,alpha_y_vox)
        beta_ridge[:,vox]=ridge_y_vox.coef_
        y_ridge[:,vox]=ridge_y_vox.predict(X_train1).T
    filename_ = rootdir_for_datafiles+ 'multiple_new5'+ str(filename_marker)+'.npz' 
    rdg = {}
    rdg['coeff'] = beta_ridge
    rdg['prediction'] = y_ridge
    rdg['alpha_pervox']=alpha_pervox
    np.savez(filename_, **rdg)
    return y_test1,y_pred


def reducedrankmultipleridge(X_train,X_val,X_test,y_train,y_val,y_test,filename_marker):
    n_vox_X=X_train.shape[1]
    n_vox_Y=y_train.shape[1]
    max_rank_B=np.min([n_vox_X,n_vox_Y])
    ranks = np.arange(1,max_rank_B+1,1,dtype=int)
    X_train1,X_val1,X_test1= scale_cv_data(X_train,X_val,X_test)
    y_train1,y_val1,y_test1= scale_cv_data(y_train,y_val,y_test)
    filename_ = rootdir_for_datafiles+ 'multiple_new5'+ str(filename_marker)+'.npz'
    rdg = np.load(filename_)
    beta_ridge=rdg['coeff']
    y_ridge=rdg['prediction']
    _, _, vh = np.linalg.svd(y_ridge)
    #save the decomposition
    filename_vh = rootdir_for_datafiles+ 'svdsolution'+ str(filename_marker)+'.npz' 
    rdg1 = {}
    rdg1['all_PC'] = vh
    np.savez(filename_vh, **rdg1)
    r_val=np.zeros([len(ranks)])
    count=0
    for r in ranks:
        B_train = (beta_ridge @ vh[:r,:].T) @ vh[:r,:]
        y_pred_val=X_val1@B_train
        r_val[count]=correlationXY(y_val1,y_pred_val)
        count=count+1
    filename_r = rootdir_for_datafiles+ 'allranks_perfold'+ str(filename_marker)+'.npz' 
    rdg1 = {}
    rdg1['allrank_r'] = r_val
    np.savez(filename_r, **rdg1)
    rank_ind=find_rank(r_val)
    rank_B=ranks[rank_ind]
    #prediction on test set
    B_train_val=(beta_ridge @ vh[:ranks[rank_ind],:].T) @ vh[:ranks[rank_ind],:]
    y_pred=X_test1@B_train_val
    return y_test1,y_pred,rank_B


def correlationXY(X,Y): #X,Y are 2D matrices, donot use on repmatted matrices
    X_tmp = (X - X.mean(axis=0)) / X.std(axis=0)
    Y_tmp = (Y - Y.mean(axis=0)) / Y.std(axis=0)
    C_mat = np.dot(Y_tmp.T, X_tmp) / Y_tmp.shape[0]
    C_diag = np.diag(C_mat)
    r_coeff= np.nanmean(C_diag)
    return r_coeff

