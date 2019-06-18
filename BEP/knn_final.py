# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:14:24 2019

@author: s163729
"""
#script for knn classifier
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.preprocessing import normalize,minmax_scale
from sklearn.model_selection import StratifiedShuffleSplit 
from pandas import DataFrame as df 
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
 
#read data automated asymmetry measurements    
data = pd.read_csv('datafinal.csv',header = 0)
data = data.dropna()

#split data in methods combined, method 1 and method 2 
X= data[['asymmetry1','asymmetry2']]
X1 = df(data['asymmetry1'])
X2 = df(data['asymmetry2'])

# ground truth 
Y= data['melanoma_truth']

#define empty lists for results 
AUC_autocombi = []
AUC_autocombi2= []
AUC_auto1 =[]
AUC_auto12 =[]
AUC_auto2=[]
AUC_auto22 = []
AUC_anno =[]
AUC_anno2 =[]
AUC_all2 =[]
AUC_all =[]

TPR_autocombi =[]
TPR2_autocombi= []
TNR_autocombi = []
TNR2_autocombi =[]
PPV_autocombi =[]
PPV2_autocombi =[]

TPR_auto1 =[]
TPR2_auto1= []
TNR_auto1 = []
TNR2_auto1 =[]
PPV_auto1 =[]
PPV2_auto1 =[]

TPR_auto2 =[]
TPR2_auto2= []
TNR_auto2 = []
TNR2_auto2 =[]
PPV_auto2 =[]
PPV2_auto2 =[]

TPR_anno =[]
TPR2_anno= []
TNR_anno = []
TNR2_anno =[]
PPV_anno =[]
PPV2_anno =[]

TPR_all =[]
TPR2_all= []
TNR_all = []
TNR2_all =[]
PPV_all =[]
PPV2_all =[]

#function for oversampling 
sm = SMOTE()

# automated asymmetry measurements combined 
for i in range(0,11):
    # split data set in training and data set 
    sss= StratifiedShuffleSplit(n_splits =5, test_size = 0.4)#, random_state=0)

    for train_index, test_index in sss.split(X,Y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
       
    #oversample training set 
    x_train,y_train = sm.fit_sample(x_train,y_train)

    #normalize training and test set 
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    #apply knn classifier with k = 5
    classifier = KNeighborsClassifier(n_neighbors=11)
    classifier.fit(x_train,y_train)
    
    #predict 
    y_pred =classifier.predict(x_test)
    y_pred2=classifier.predict(x_train)
    
    #accuracy of training and test set 
    accuracy = accuracy_score(y_test, y_pred)
    accuracy2= accuracy_score(y_train,y_pred2)
    
    #AUC of training and test set 
    fpr= dict()
    tpr= dict()
    roc_auc =dict()
    fpr,tpr, _ = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr, tpr)
    AUC_autocombi.append(float(roc_auc))
    
    fpr2= dict()
    tpr2= dict()
    roc_auc2 =dict()
    fpr2,tpr2, _ = roc_curve(y_train,y_pred2)
    roc_auc2 = auc(fpr2, tpr2)
    AUC_autocombi2.append(float(roc_auc2))
    
    #measuring sensitivity, specificity and precision 
    conf = confusion_matrix(y_test, y_pred)
    conf2 = confusion_matrix(y_train, y_pred2)
    
    TN = conf[0][0]
    TN2= conf2[0][0]
    FN = conf [1][0]
    FN2 = conf2[1][0]
    TP = conf[1][1]
    TP2 =conf2[1][1]
    FP =conf[0][1]
    FP2 = conf2[0][1]
    TPR = TP/(TP+FN)
    TPR_autocombi.append(float(TPR))
    TPR2 = TP2/(TP2+FN2)
    TPR2_autocombi.append(float(TPR2))
    TNR = TN/(TN+FP)
    TNR_autocombi.append(float(TNR))
    TNR2 = TN2/(TN2+FP2)
    TNR2_autocombi.append(float(TNR2))
    PPV = TP/(TP+FP)
    PPV_autocombi.append(float(PPV))
    PPV2 = TP2/(TP2+FP2)
    PPV2_autocombi.append(float(PPV2))
       
    

#method 1     
for i in range(0,11):
    
    #split data in test and training set 
    sss= StratifiedShuffleSplit(n_splits =5, test_size = 0.4)#, random_state=0)

    for train_index, test_index in sss.split(X1,Y):
        x_train, x_test = X1.iloc[train_index], X1.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    #oversample training set 
    x_train,y_train = sm.fit_sample(x_train,y_train)

    #normalize training and test set 
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    #apply knn classifier with k =5 
    classifier = KNeighborsClassifier(n_neighbors=11)
    classifier.fit(x_train,y_train)

    #predict
    y_pred =classifier.predict(x_test)
    y_pred2=classifier.predict(x_train)
    
    #accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy2=accuracy_score(y_train,y_pred2)

    #AUC of test and training set 
    fpr= dict()
    tpr= dict()
    roc_auc =dict()
    fpr,tpr, _ = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr, tpr)
    AUC_auto1.append(float(roc_auc))
    
    fpr2= dict()
    tpr2= dict()
    roc_auc2 =dict()
    fpr2,tpr2, _ = roc_curve(y_train,y_pred2)
    roc_auc2 = auc(fpr2, tpr2)
    AUC_auto12.append(float(roc_auc2))
    
    #measuring sensitivity, specificity and precision 
    conf = confusion_matrix(y_test, y_pred)
    conf2 = confusion_matrix(y_train, y_pred2)
    
    TN = conf[0][0]
    TN2= conf2[0][0]
    FN = conf [1][0]
    FN2 = conf2[1][0]
    TP = conf[1][1]
    TP2 =conf2[1][1]
    FP =conf[0][1]
    FP2 = conf2[0][1]
    TPR = TP/(TP+FN)
    TPR2 = TP2/(TP2+FN2)
    TNR = TN/(TN+FP)
    TNR2 = TN2/(TN2+FP2)
    PPV = TP/(TP+FP)
    PPV2 = TP2/(TP2+FP2)
    
    TPR_auto1.append(float(TPR))
    TPR2_auto1.append(float(TPR))
    TNR_auto1.append(float(TNR))
    TNR2_auto1.append(float(TNR2))
    PPV_auto1.append(float(PPV))
    PPV2_auto1.append(float(PPV2))

#method 2     
for i in range(0,11):
    #split data set in training and test set 
    sss= StratifiedShuffleSplit(n_splits =5, test_size = 0.4)#, random_state=0)

    for train_index, test_index in sss.split(X2,Y):
        x_train, x_test = X2.iloc[train_index], X2.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    
    #oversample training set 
    x_train, y_train = sm.fit_sample(x_train,y_train)

    #normalize test and training set 
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    #apply knn classifier with k = 5 
    classifier = KNeighborsClassifier(n_neighbors=11)
    classifier.fit(x_train,y_train)

    #predict classes
    y_pred =classifier.predict(x_test)
    y_pred2 = classifier.predict(x_train)
    
    #accuracy 
    accuracy = accuracy_score(y_test, y_pred)
    accuracy2= accuracy_score(y_train, y_pred2)
    
    #AUC of test and training set 
    fpr= dict()
    tpr= dict()
    roc_auc =dict()
    fpr,tpr, _ = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr, tpr)
    AUC_auto2.append(float(roc_auc))
    
    fpr2= dict()
    tpr2= dict()
    roc_auc2 =dict()
    fpr2,tpr2, _ = roc_curve(y_train,y_pred2)
    roc_auc2 = auc(fpr2, tpr2)
    AUC_auto22.append(float(roc_auc2))

    #measuring sensitivity, specificity and precision 
    conf = confusion_matrix(y_test, y_pred)
    conf2 = confusion_matrix(y_train, y_pred2)
    
    TN = conf[0][0]
    TN2= conf2[0][0]
    FN = conf [1][0]
    FN2 = conf2[1][0]
    TP = conf[1][1]
    TP2 =conf2[1][1]
    FP =conf[0][1]
    FP2 = conf2[0][1]
    TPR = TP/(TP+FN)
    TPR2 = TP2/(TP2+FN2)
    TNR = TN/(TN+FP)
    TNR2 = TN2/(TN2+FP2)
    PPV = TP/(TP+FP)
    PPV2 = TP2/(TP2+FP2)
    
    TPR_auto2.append(float(TPR))
    TPR2_auto2.append(float(TPR))
    TNR_auto2.append(float(TNR))
    TNR2_auto2.append(float(TNR2))
    PPV_auto2.append(float(PPV))
    PPV2_auto2.append(float(PPV2))

#read in data from annotations per group 
data1 = pd.read_excel('group1.xlsx', header = 0)
data1 = data1.iloc[:,:-2]
data1 = data1.dropna()  
asym1 = data1[['Asymmetry_1_1', 'Asymmetry_1_2', 'Asymmetry_1_3']] 
asym1= asym1.replace([0,2],[2,0])

data2 = pd.read_excel('group2.xlsx', header = 0)
data2 = data2.iloc[:,:]
data2 = data2.dropna()
asym2 = data2[['Asymmetry_2_1','Asymmetry_2_2', 'Asymmetry_2_3']]
asym2 = asym2.replace([1,2,3,4,5],[5,4,3,2,1])

data3 = pd.read_excel('group3.xlsx', header = 0)
data3 = data3.iloc[:,:-2]
data3 = data3.dropna()
asym3= data3[['Asymmetry_3_1','Asymmetry_3_2', 'Asymmetry_3_3']]
asym3 = asym3.replace([0,1],[1,0])

data4 = pd.read_excel('group4.xlsx', header = 0)
data4 = data4.iloc[:,:-2]
data4 = data4.dropna()
asym4 = data4[['Asymmetry_4_1', 'Asymmetry_4_3','Asymmetry_4_5']]
asym4= asym4.replace([0,2],[2,0])

data5 = pd.read_excel('group5.xlsx', header = 0)
data5 = data5.iloc[:,:-2]
asym5 = data5[['Asymmetry_5_1', 'Asymmetry_5_2', 'Asymmetry_5_3']]
asym5= asym5.replace([0,2],[2,0])
#data5 = data5.dropna()

data6 = pd.read_excel('group6.xlsx', header = 0)
data6 = data6.iloc[:,:-2]
data6 = data6.dropna()
asym6 = data6[['Asymmetry_6_1', 'Asymmetry_6_2', 'Asymmetry_6_3']]
asym6= asym6.replace([0,2],[2,0])

data7 = pd.read_excel('group7.xlsx', header = 0)
data7 = data7.iloc[:,:-2]
data7 = data7.dropna()
data7 = data7.reset_index()
asym7= data7[['Asymmetry_7_1','Asymmetry_7_2','Asymmetry_7_3','Asymmetry_7_4','Asymmetry_7_5','Asymmetry_7_6']]
asym7= asym7.replace([0,2],[2,0])

data8 = pd.read_excel('group8.xlsx', header = 0)
data8 = data8.dropna()
asym8 = data8[['Asymmetry_8_1','Asymmetry_8_2','Asymmetry_8_3']]
asym8= asym8.replace([0,2],[2,0])

#scale each feature per group and use the means of these features
a_1 = ((minmax_scale(asym1['Asymmetry_1_1'])+ minmax_scale(asym1['Asymmetry_1_2']) + minmax_scale(asym1['Asymmetry_1_3']))/3.0) #columns = ['Asymmetry_1'])
a_2 = ((minmax_scale(asym2['Asymmetry_2_1'])+ minmax_scale(asym2['Asymmetry_2_2']) + minmax_scale(asym2['Asymmetry_2_3']))/3.0) #columns = ['Asymmetry_2'])
a_3 = ((minmax_scale(asym3['Asymmetry_3_1'])+ minmax_scale(asym3['Asymmetry_3_2']) + minmax_scale(asym3['Asymmetry_3_3']))/3.0) #columns = ['Asymmetry_3'])
a_4 = ((minmax_scale(asym4['Asymmetry_4_1'])+ minmax_scale(asym4['Asymmetry_4_3']) + minmax_scale(asym4['Asymmetry_4_5']))/3.0) #columns = ['Asymmetry_4'])
a_5 = ((minmax_scale(asym5['Asymmetry_5_1'])+ minmax_scale(asym5['Asymmetry_5_2']) + minmax_scale(asym5['Asymmetry_5_3']))/3.0) #columns = ['Asymmetry_5'])
a_6 = ((minmax_scale(asym6['Asymmetry_6_1'])+ minmax_scale(asym6['Asymmetry_6_2']) + minmax_scale(asym6['Asymmetry_6_3']))/3.0) #columns = ['Asymmetry_6'])
a_7 = ((minmax_scale(asym7['Asymmetry_7_1'])+ minmax_scale(asym7['Asymmetry_7_2']) + minmax_scale(asym7['Asymmetry_7_3'])+ 
        minmax_scale(asym7['Asymmetry_7_4']) + minmax_scale(asym7['Asymmetry_7_5']) + minmax_scale(asym7['Asymmetry_7_6']))/6.0) #columns = ['Asymmetry_7'])
a_8 = ((minmax_scale(asym8['Asymmetry_8_1'])+ minmax_scale(asym8['Asymmetry_8_2']) +minmax_scale(asym8['Asymmetry_8_3']))/3.0) #columns = ['Asymmetry_8'])

#combine all asymmetry annotations 
asymm_label=np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8))
asym_label = df(asymm_label)
asym_label = asym_label.reset_index()
asym_label = df(asym_label.iloc[:,1])

# combine groundtruth per group 
truth =  pd.concat([data1['Melanoma'], data2['Melanoma'], data3['Melanoma'], data4['Melanoma'], data5['Melanoma'], data6['Melanoma'],data7['Melanoma'], data8['Melanoma']],axis =0, sort = False)
truth = truth.reset_index()
truth = df(truth.iloc[:,1])

#annotations 
for i in range(0,11):
    #split data set in training and test set 
    sss= StratifiedShuffleSplit(n_splits =5, test_size = 0.4)#, random_state=0)
    for train_index, test_index in sss.split(asym_label,truth):
        x_train, x_test = asym_label.iloc[train_index], asym_label.iloc[test_index]
        y_train, y_test = truth.iloc[train_index], truth.iloc[test_index]
    
    #oversample training set
    x_train,y_train = sm.fit_sample(x_train,y_train)
    
    #normalize training and test set 
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    #apply knn classifier with k =5 
    classifier = KNeighborsClassifier(n_neighbors=11)
    classifier.fit(x_train,y_train)
    
    #predict classes
    y_pred =classifier.predict(x_test)
    y_pred2= classifier.predict(x_train)

    #accuracy 
    accuracy = accuracy_score(y_test, y_pred)
    accuracy2= accuracy_score(y_train, y_pred2)
    
    #AUC of test and training set 
    fpr= dict()
    tpr= dict()
    roc_auc =dict()
    fpr,tpr, _ = roc_curve(y_test,y_pred)
    roc_auc = auc(fpr, tpr)
    AUC_anno.append(float(roc_auc))
    
    fpr2= dict()
    tpr2= dict()
    roc_auc2 =dict()
    fpr2,tpr2, _ = roc_curve(y_train,y_pred2)
    roc_auc2 = auc(fpr2, tpr2)
    AUC_anno2.append(float(roc_auc2))
    
    #measure sensitivity, specificity and precision 
    conf = confusion_matrix(y_test, y_pred)
    conf2 = confusion_matrix(y_train, y_pred2)
    
    TN = conf[0][0]
    TN2= conf2[0][0]
    FN = conf [1][0]
    FN2 = conf2[1][0]
    TP = conf[1][1]
    TP2 =conf2[1][1]
    FP =conf[0][1]
    FP2 = conf2[0][1]
    TPR = TP/(TP+FN)
    TPR2 = TP2/(TP2+FN2)
    TNR = TN/(TN+FP)
    TNR2 = TN2/(TN2+FP2)
    PPV = TP/(TP+FP)
    PPV2 = TP2/(TP2+FP2)
    
    TPR_anno.append(float(TPR))
    TPR2_anno.append(float(TPR))
    TNR_anno.append(float(TNR))
    TNR2_anno.append(float(TNR2))
    PPV_anno.append(float(PPV))
    PPV2_anno.append(float(PPV2))


#read in data from all students 
#asymmetry 
data_a = pd.read_csv('datafinal.csv',header=0)
#border
data_b = pd.read_csv('data_audrey.csv',header =0)
#color
data_c = pd.read_csv('data_final_Emiel.csv',header =0)

#combine data sets 
frames = [data_a,data_b,data_c]
data_all = pd.concat(frames, axis =1, sort = False)
#split features from groundtruth 
data_all = data_all[['asymmetry1','asymmetry2','compactness','convex','abruptness','p1','p2','p3', 'p4','p5','p6','p7','p8','p9','melanoma_truth']]
data_all = data_all.mask(data_all.eq('None')).dropna()
data_all = data_all.mask(data_all.eq('inf')).dropna()
data_all = data_all.reset_index()

#features (X_all) and groundtruth (Y_all)
X_all = data_all[['asymmetry1','asymmetry2','compactness','convex','abruptness','p1','p2','p3', 'p4','p5','p6','p7','p8','p9']]
Y_all= data_all['melanoma_truth']

#all ABC features 
for i in range(0,11):
    #split data set in training and test set 
    sss= StratifiedShuffleSplit(n_splits =5, test_size = 0.2) #, random_state=0)
    for train_index, test_index in sss.split(X_all,Y_all):
        x_train, x_test = X_all.iloc[train_index], X_all.iloc[test_index]
        y_train, y_test = Y_all.iloc[train_index], Y_all.iloc[test_index] 
    #oversample training set 
    x_train,y_train = sm.fit_sample(x_train,y_train)
    #normalize training and test set 
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    #apply knn classifier with k=5 
    classifier = KNeighborsClassifier(n_neighbors=11)
    classifier.fit(x_train,y_train)

    #predict classes
    predictions =classifier.predict(x_test)
    predictions2=classifier.predict(x_train)

    #accuracy 
    accuracy = accuracy_score(y_test, predictions)
    accuracy2=accuracy_score(y_train,predictions2)

    #AUC of test and training set 
    fpr= dict()
    tpr= dict()
    roc_auc =dict()
    fpr,tpr, _ = roc_curve(y_test,predictions)
    roc_auc = auc(fpr, tpr)
    AUC_all.append(float(roc_auc))
    
    fpr2= dict()
    tpr2= dict()
    roc_auc2 =dict()
    fpr2,tpr2, _ = roc_curve(y_train,predictions2)
    roc_auc2 = auc(fpr2, tpr2)
    AUC_all2.append(float(roc_auc2))
    
    #measure sensitivity, specificity and precision 
    conf = confusion_matrix(y_test, predictions)
    conf2 = confusion_matrix(y_train, predictions2)
        
    TN = conf[0][0]
    TN2= conf2[0][0]
    FN = conf [1][0]
    FN2 = conf2[1][0]
    TP = conf[1][1]
    TP2 =conf2[1][1]
    FP =conf[0][1]
    FP2 = conf2[0][1]
    TPR = TP/(TP+FN)
    TPR2 = TP2/(TP2+FN2)
    TNR = TN/(TN+FP)
    TNR2 = TN2/(TN2+FP2)
    PPV = TP/(TP+FP)
    PPV2 = TP2/(TP2+FP2)
    
    TPR_all.append(float(TPR))
    TPR2_all.append(float(TPR))
    TNR_all.append(float(TNR))
    TNR2_all.append(float(TNR2))
    PPV_all.append(float(PPV))
    PPV2_all.append(float(PPV2))



#Annotations test and train 
#measure means 
print('mean AUC annotations test= '+ str(np.mean(AUC_anno)))
print('mean sensitivity annotations test= '+ str(np.mean(TPR_anno)))
print('mean specificity annotations test= '+ str(np.mean(TNR_anno)))
print('mean precision annotations test= '+ str(np.mean(PPV_anno)))

print('mean AUC annotations train= '+ str(np.mean(AUC_anno2)))
print('mean sensitivity annotations train= '+ str(np.mean(TPR2_anno)))
print('mean specificity annotations train= '+ str(np.mean(TNR2_anno)))
print('mean precision annotations train= '+ str(np.mean(PPV2_anno)))

#Autocombi test and train 
#measure means 
print('mean AUC method 1 and 2 test= '+ str(np.mean(AUC_autocombi)))
print('mean sensitivity method 1 and 2 test= '+ str(np.mean(TPR_autocombi)))
print('mean specificity method 1 and 2 test= '+ str(np.mean(TNR_autocombi)))
print('mean precision method 1 and 2 test= '+ str(np.mean(PPV_autocombi)))

print('mean AUC method 1 and 2 train= '+ str(np.mean(AUC_autocombi2)))
print('mean sensitivity method 1 and 2 train= '+ str(np.mean(TPR2_autocombi)))
print('mean specificity method 1 and 2 train= '+ str(np.mean(TNR2_autocombi)))
print('mean precision method 1 and 2 train= '+ str(np.mean(PPV2_autocombi)))

#Method 1 test and train 
#measure means 
print('mean AUC method 1 test= '+str(np.mean(AUC_auto1)))
print('mean sensitivity method 1 test= '+ str(np.mean(TPR_auto1)))
print('mean specificity method 1 test= '+ str(np.mean(TNR_auto1)))
print('mean precision method 1 test= '+ str(np.mean(PPV_auto1)))

print('mean AUC method 1 train= '+ str(np.mean(AUC_auto12)))
print('mean sensitivity method 1 train= '+ str(np.mean(TPR2_auto1)))
print('mean specificity method 1 train= '+ str(np.mean(TNR2_auto1)))
print('mean precision method 1 train= '+ str(np.mean(PPV2_auto1)))

#Method 2 test and train
#measure means  
print('mean AUC method 2 test= '+str(np.mean(AUC_auto2)))
print('mean sensitivity method 2 test= '+ str(np.mean(TPR_auto2)))
print('mean specificity method 2 test= '+ str(np.mean(TNR_auto2)))
print('mean precision method 2 test= '+ str(np.mean(PPV_auto2)))

print('mean AUC method 2 train= '+ str(np.mean(AUC_auto22)))
print('mean sensitivity method 2 train= '+ str(np.mean(TPR2_auto2)))
print('mean specificity method 2 train= '+ str(np.mean(TNR2_auto2)))
print('mean precision method 2 train= '+ str(np.mean(PPV2_auto2)))

#All 
#measure means 
print('mean AUC ABC features test= '+ str(np.mean(AUC_all)))
print('mean sensitivity ABC features test= '+ str(np.mean(TPR_all)))
print('mean specificity ABC features test= '+ str(np.mean(TNR_all)))
print('mean precision ABC features test= '+ str(np.mean(PPV_all)))

print('mean AUC ABC features train= '+ str(np.mean(AUC_all2)))
print('mean sensitivity ABC features train= '+ str(np.mean(TPR2_all)))
print('mean specificity ABC features train= '+ str(np.mean(TNR2_all)))
print('mean precision ABC features train= '+ str(np.mean(PPV2_all)))
 

#make boxplots different methods 
my_dict = {'Annotations Asymmetry': AUC_anno, 'Method 1 and 2 ': AUC_autocombi, 'Method 1': AUC_auto1, 'Method 2': AUC_auto2, 'All features (ABC)':AUC_all}
fig, ax = plt.subplots()
ax.boxplot(my_dict.values())
ax.set_xticklabels(my_dict.keys(), rotation =45, fontsize =10) 
plt.ylabel('AUC') 
plt.title('AUC k-Nearest Neighbor, k=5') 

