# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:58:38 2019

@author: s162152
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,roc_curve, auc, confusion_matrix
import numpy as np
from sklearn import preprocessing
from pathtofiles import *

def classify_visual(): 
    ###Import data of different groups
    df_1 = pd.read_excel(pathtofiles()+'groups/group01.xlsx')
    df_2 = pd.read_excel(pathtofiles()+'groups/group02.xlsx')
    df_3 = pd.read_excel(pathtofiles()+'groups/group03.xlsx')
    df_4 = pd.read_excel(pathtofiles()+'groups/group04.xlsx')
    df_5 = pd.read_excel(pathtofiles()+'groups/group05.xlsx')
    df_6 = pd.read_excel(pathtofiles()+'groups/group06.xlsx')
    df_7 = pd.read_excel(pathtofiles()+'groups/group07.xlsx')  
    df_3 = df_3.reset_index()
    df_5 = df_5.loc[:,['ID','Border_5_1','Border_5_2','Border_5_3']].dropna()
    df_5 = df_5.reset_index()
    df_7 = df_7.dropna()
    df_7 = df_7.reset_index()
    
    ##Scale and average the visual scores for border to be able to merge them
    a_1 = (preprocessing.scale(df_1['Border_1_1']) + preprocessing.scale(df_1['Border_1_2']) + preprocessing.scale(df_1['Border_1_3'])) / 3.0
    a_2 = (preprocessing.scale(df_2['Rand onregelmatigheid']) + preprocessing.scale(df_2['Unnamed: 6']) + preprocessing.scale(df_2['Unnamed: 7'])) / 3.0
    a_3 = (preprocessing.scale(df_3['Border_3_1']) + preprocessing.scale(df_3['Border_3_2']) + preprocessing.scale(df_3['Border_3_3'])) / 3.0
    a_4 = (preprocessing.scale(df_4['Border_4_1']) + preprocessing.scale(df_4['Border_4_3']) + preprocessing.scale(df_4['Border_4_5'])) / 3.0
    a_5 = (preprocessing.scale(df_5['Border_5_1']) + preprocessing.scale(df_5['Border_5_2']) + preprocessing.scale(df_5['Border_5_3'])) / 3.0
    a_6 = (preprocessing.scale(df_6['Border_6_1']) + preprocessing.scale(df_6['Border_6_2']) + preprocessing.scale(df_6['Border_6_3'])) / 3.0
    a_7 = (preprocessing.scale(df_7['Border_7_1']) + preprocessing.scale(df_7['Border_7_2']) + preprocessing.scale(df_7['Border_7_3']) + preprocessing.scale(df_7['Border_7_4']) + preprocessing.scale(df_7['Border_7_5']) + preprocessing.scale(df_7['Border_7_6'])) / 6.0
    
    ##merge the visual scores for border and reshape the labels (-1,1), because it is one feature
    group_label=np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7))
    group_label=group_label.reshape(-1,1)
    group_id=np.concatenate((df_1['ID'],df_2['Afbeelding'],df_3['Unnamed: 0'],df_4['ID'],df_5['ID'],df_6['ID'],df_7['ID']))
    
    ##Load the groundtruth
    df = pd.read_csv(pathtofiles()+'csvfiles/GroundTruth.csv')
    class_label=df['melanoma']
    class_id=df['image_id']
    
    ##Find the labels from the groups in the groundtruth 
    true_group_label=[]
    for i in range(len(group_id)):
        for j in range(len(class_id)):
            if group_id[i]==class_id[j]:
                true_group_label.append(class_label[j])
    
    ##Scale the dataset with mean unit variance
    scaler = StandardScaler()
    group_label=scaler.fit_transform(group_label)
    
    true_group_label=np.array(true_group_label, np.float64)
    
    ## Create an array of the group labels
    X = group_label 
    ## Create an array of the groundtruth of those group labels
    y = true_group_label 

    scores_LR_vis= []
    scores_svm_vis=[]
    
    ##Append 10-fold cross validation
    cv = KFold(n_splits=9, random_state=42, shuffle=True)
    for train_index, test_index in cv.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        
        
        ###Logistic regression
        ## Initialize classifier for imbalanced dataset
        LR = LogisticRegression(solver='lbfgs', class_weight='balanced')
        ## Train classifier
        model = LR.fit(X_train, y_train)
        ## Make predictions
        preds = LR.predict(X_test)
        
        ##Determine the AUC scores and append them to a list
        fpr, tpr, thresholds = roc_curve(y_test, preds)
       
        AUC_score_LR=auc(fpr, tpr)
        if np.isnan(AUC_score_LR):
            continue
        else:
            scores_LR_vis.append(AUC_score_LR)
    
    
        ###SVM 
        ## Initialize classifier for imbalanced dataset
        clf = svm.SVC(class_weight='balanced')
        ## Train classifier
        clf.fit(X_train, y_train)
        ## Make predictions
        prd=clf.predict(X_test)
        
        ##Determine the AUC scores and append them to a list
        fpr_svm,tpr_svm, thresholds=roc_curve(y_test,prd)    
        
        AUC_score_svm=auc(fpr_svm,tpr_svm)
            
        if np.isnan(AUC_score_svm):
            continue
        else:
            scores_svm_vis.append(AUC_score_svm)
    
    
    return scores_svm_vis, scores_LR_vis


