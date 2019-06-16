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
from pathtofiles import *
import numpy as np
from sklearn import preprocessing


def classify_automatic(df1,option,mask):
    ##Delete the rows that contain "none" or "inf"values 
    df1=df1.mask(df1.eq('None')).dropna()
    df1=df1.mask(df1.eq('inf')).dropna()
    
    ##Load the groundtruth
    df = pd.read_csv(pathtofiles()+'csvfiles/GroundTruth.csv')
    class_label=df['melanoma']
    class_id=df['image_id']
    
    ##Add groundtruth to the dataset
    df1['true']=class_label
    
    ##If you want the data of all 2000 images with border, asymmetry and color, Load those datasets as well
    if option=="2000all":
        df2=pd.read_csv(pathtofiles()+'csvfiles/data_lijst_2000.csv')
        df3=pd.read_csv(pathtofiles()+'csvfiles/data_final.csv')
        df4=pd.read_csv(pathtofiles()+'csvfiles/data_final2.csv')

        df1['true']=class_label
        df1['asymmetry1']=df3['asymmetry']
        df1['asymmetry2']=df4['asymmetry']
        df1['p1']=df2['p1']
        df1['p2']=df2['p2']
        df1['p3']=df2['p3']
        df1['p4']=df2['p4']
        df1['p5']=df2['p5']
        df1['p6']=df2['p6']
        df1['p7']=df2['p7']
        df1['p8']=df2['p8']
        df1['p9']=df2['p9']
        data=df1
    ##If you want to compare the border methods with the visual scores of the border load the visual scores and only keep the data of the images with visual scores
    elif option=="visual":
        df_1 = pd.read_excel(pathtofiles()+'groups/group01.xlsx')
        df_2 = pd.read_excel(pathtofiles()+'groups/group02.xlsx')
        df_3 = pd.read_excel(pathtofiles()+'groups/group03.xlsx')
        df_4 = pd.read_excel(pathtofiles()+'groups/group04.xlsx')
        df_5 = pd.read_excel(pathtofiles()+'groups/group05.xlsx')
        df_6 = pd.read_excel(pathtofiles()+'groups/group06.xlsx')
        df_7 = pd.read_excel(pathtofiles()+'groups/group07.xlsx')
        df_3 = df_3.reset_index()
        df_7 = df_7.dropna()
        df_7 = df_7.reset_index()
        
        
        a_1 = (preprocessing.scale(df_1['Border_1_1']) + preprocessing.scale(df_1['Border_1_2']) + preprocessing.scale(df_1['Border_1_3'])) / 3.0
        a_2 = (preprocessing.scale(df_2['Rand onregelmatigheid']) + preprocessing.scale(df_2['Unnamed: 6']) + preprocessing.scale(df_2['Unnamed: 7'])) / 3.0
        a_3 = (preprocessing.scale(df_3['Border_3_1']) + preprocessing.scale(df_3['Border_3_2']) + preprocessing.scale(df_3['Border_3_3'])) / 3.0
        a_4 = (preprocessing.scale(df_4['Border_4_1']) + preprocessing.scale(df_4['Border_4_3']) + preprocessing.scale(df_4['Border_4_5'])) / 3.0
        a_5 = (preprocessing.scale(df_5['Border_5_1']) + preprocessing.scale(df_5['Border_5_2']) + preprocessing.scale(df_5['Border_5_3'])) / 3.0
        a_6 = (preprocessing.scale(df_6['Border_6_1']) + preprocessing.scale(df_6['Border_6_2']) + preprocessing.scale(df_6['Border_6_3'])) / 3.0
        a_7 = (preprocessing.scale(df_7['Border_7_1']) + preprocessing.scale(df_7['Border_7_2']) + preprocessing.scale(df_7['Border_7_3']) + preprocessing.scale(df_7['Border_7_4']) + preprocessing.scale(df_7['Border_7_5']) + preprocessing.scale(df_7['Border_7_6'])) / 6.0
        
        #combine visual scores and image IDs of the different groups 
        group_label=np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7))
        group_label=group_label.reshape(-1,1)
        group_id=np.concatenate((df_1['ID'],df_2['Afbeelding'],df_3['Unnamed: 0'],df_4['ID'],df_5['ID'],df_6['ID'],df_7['ID']))

        true_id=[]
        #
        for i in range(len(group_id)):
            for j in range(len(class_id)):
                if group_id[i]==class_id[j]:
                    true_id.append(class_id[j])
        ###To keep the data of the images with visual scores the code has to be altered for which mask is used
        for i in range(len(true_id)):
            if mask=="created":
                true_id[i]=str(true_id[i])+str('.jpg')
            elif mask=="given":
                true_id[i]=str(true_id[i])+str('_segmentation.png')
        data=df1[df1['image'].isin(true_id)]
    
    ##If you just want to compare the border methods of the 2000 images, keep the data the same as the loaded data
    elif option=="2000":
        data=df1
    
    ##Decide whether the boxplots need to be of the border methods or also of the asymmetry and color features
    border_id=data['image']
    if option=="visual" or option=="2000":
        variations=[['compactness'],['convex'],['abruptness'],['compactness','convex','abruptness']]
    elif option=="2000all":
        variations=[['compactness','convex','abruptness'],['asymmetry1','asymmetry2'],['p1','p2','p3','p4','p5','p6','p7','p8','p9'],['compactness','convex','abruptness','asymmetry1','asymmetry2','p1','p2','p3','p4','p5','p6','p7','p8','p9']]
    for i in range(len(variations)):
        border_label=data.loc[:,variations[i]]
        true_label=data['true']
        border_label=border_label.reset_index(drop=True)
        true_label=true_label.reset_index(drop=True)
        border_label=np.array(border_label,np.float64)
        
        ##One has to reshape the data if it only contains one feature
        if option=="visual" or option=="2000":
            if i==0 or i==1 or i==2:
                border_label=border_label.reshape(-1,1)
        
        ## Scale the data with unit mean variance
        scaler = StandardScaler()
        border_label=scaler.fit_transform(border_label)

        
        ## Create an array of the border labels
        X = border_label 
        ## Create an array of the groundtruth of those border labels
        y = true_label 

    
        scores_LR= []
        scores_svm=[]
        
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
                scores_LR.append(AUC_score_LR)
            
        
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
                scores_svm.append(AUC_score_svm)
          
            ##Save the scores of the variations separately 
            if i==0:
                scores_LR_1=scores_LR
                scores_svm_1=scores_svm
            elif i==1:
                scores_LR_2=scores_LR
                scores_svm_2=scores_svm
            elif i==2:
                scores_LR_3=scores_LR
                scores_svm_3=scores_svm
            else:
                scores_LR_combined=scores_LR
                scores_svm_combined=scores_svm
    return scores_LR_1,scores_svm_1, scores_LR_2,scores_svm_2,scores_LR_3,scores_svm_3,scores_LR_combined,scores_svm_combined

        