
# -*- coding: utf-8 -*-
"""
Logistic Regression for CrowdAnnotations
Groups 01,02,04,06,07,08 included
in line 86 possible to select features ABC.
@author: s151385
""" 
    
###
import pandas as pd
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from config import MAIN_PATH as MAIN_PATH
from config import GROUP_PATH as GROUP_PATH

def ReadGroup(group):
    if group == 1:
        classes_C = ['Color_1_1','Color_1_2','Color_1_3']
        classes_A = ['Asymmetry_1_1','Asymmetry_1_2','Asymmetry_1_3']
        classes_B = ['Border_1_1','Border_1_2','Border_1_3']
        classes_all = ['ID'] + classes_C + classes_A + classes_B
    elif group == 2:
        classes_C = ['Color_2_1','Color_2_2','Color_2_3']
        classes_A = ['Asymmetry_2_1','Asymmetry_2_2','Asymmetry_2_3']
        classes_B = ['Border_2_1','Border_2_2','Border_2_3']
        classes_all = ['ID'] + classes_C + classes_A + classes_B
    elif group == 4: 
        classes_C = ['Color_4_2','Color_4_4','Color_4_6']
        classes_A = ['Asymmetry_4_1','Asymmetry_4_3','Asymmetry_4_5']
        classes_B = ['Border_4_1','Border_4_3','Border_4_5']
        classes_all = ['ID'] + classes_C + classes_A + classes_B
    elif group == 6:
        classes_C = ['Color_6_1','Color_6_2','Color_6_3']
        classes_A = ['Asymmetry_6_1','Asymmetry_6_2','Asymmetry_6_3']
        classes_B = ['Border_6_1','Border_6_2','Border_6_3']
        classes_all = ['ID'] + classes_C + classes_A + classes_B
    elif group == 7:
        classes_C = ['Color_7_1','Color_7_2','Color_7_3','Color_7_4','Color_7_5','Color_7_6']
        classes_A = ['Asymmetry_7_1','Asymmetry_7_2','Asymmetry_7_3','Asymmetry_7_4','Asymmetry_7_5','Asymmetry_7_6']
        classes_B = ['Border_7_1','Border_7_2','Border_7_3','Border_7_4','Border_7_5','Border_7_6']
        classes_all = ['ID'] + classes_C + classes_A + classes_B
    elif group == 8:
        classes_C = ['Color_8_1','Color_8_2','Color_8_3']
        classes_A = ['Asymmetry_8_1','Asymmetry_8_2','Asymmetry_8_3']
        classes_B = ['Border_8_1','Border_8_2','Border_8_3']
        classes_all = ['ID'] + classes_C + classes_A + classes_B
    
    ### imread file
    os.chdir(GROUP_PATH)
    data_raw = pd.read_excel('group0'+str(group)+'.xlsx')
    data = data_raw[classes_all]
    data_clean = data.dropna(axis='rows')
    ID = data_clean['ID']
    ID= np.array(ID)
    
    melanoom_truth = []
    image_names = []  
    os.chdir(MAIN_PATH)
    with open('ISIC-2017_Training_Part3_GroundTruth.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader,None) #skip headers 
        for lines in reader:
            image_names.append(str(lines[0]))
            melanoom_truth.append(str(lines[1]))
    
    C = data_clean[classes_C]
    C_np = np.array(C)
    C_mean = np.zeros([len(C_np),1])
    truth = np.zeros([len(C_np),1])
    for i in range(0,len(C_np)):
        C_mean[i] = np.mean(C_np[i,:])
        truth[i] = melanoom_truth[image_names.index(ID[i])]
  
    A = data_clean[classes_A]
    A_np = np.array(A)
    A_mean = np.zeros([len(A_np),1])
    for i in range(0,len(C_np)):
        A_mean[i] = np.mean(A_np[i,:])
    
        
    B = data_clean[classes_B]
    B_np = np.array(B)
    B_mean = np.zeros([len(B_np),1])
    for i in range(0,len(B_np)):
        B_mean[i] = np.mean(B_np[i,:])
    output = np.concatenate([A_mean,B_mean,C_mean,truth],axis=1)
    scaler = MinMaxScaler()
    scaler.fit(output)
    output_ready = scaler.transform(output)
    return(output_ready)

def loadData() :
    data01 = ReadGroup(1)
    data02 = ReadGroup(2)
    data04 = ReadGroup(4)
    data06 = ReadGroup(6)
    data07 = ReadGroup(7)
    data08 = ReadGroup(8)
    data_all = np.concatenate([data01, data02, data04, data06, data07, data08])
    return(data_all)

def splitData(data_all):
    x_data = data_all[:,0:-1]
    y_data = data_all[:,-1]
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33, random_state=None, shuffle = True)
    return(x_train,y_train,x_test,y_test)

def logReg(x_train,y_train,x_test,y_test):
    # logistic regression
    from CM_ROC import ROC
    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression(max_iter = 1000000,dual=False, class_weight='balanced',
                                      solver='liblinear')
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test,y_test)
    #CM(y_test,predictions)
    roc_auc = ROC(y_test,predictions)
    return(roc_auc)
    
def main():
    data_all = loadData()
    roc_new = []
    for i in range(0,100):
        x_train,y_train,x_test,y_test = splitData(data_all)
        roc_new.append(logReg(x_train,y_train,x_test,y_test))
    return(roc_new)
 
roc_crowd_ABC = main()
plt.boxplot(roc_crowd_ABC)
