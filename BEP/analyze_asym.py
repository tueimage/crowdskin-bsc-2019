# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:36:06 2019

@author: s163729
"""

def analyze_asym(data):
    #analysis of the asymmetry data 
    #data = datafinal --> asymmetry method 1 and 2 
    from functions import meanANDstd,shapiro, ztest,RainCloud 
    
    #Shapiro-Wilk test 
    shapiro(data)
    
    #z-test
    ztest(data)
    
    #determine the mean and standard deviation 
    meanANDstd(data)
   
    #Raincloud plot method 1 and 2 
    RainCloud(data)
    
    #Raincloud plot annotations 
    import pandas as pd 
    from sklearn.preprocessing import minmax_scale
    import numpy as np
    from pandas import DataFrame as df
    #read data annotations 
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
    
    #scale annotations 
    a_1 = ((minmax_scale(asym1['Asymmetry_1_1'])+ minmax_scale(asym1['Asymmetry_1_2']) + minmax_scale(asym1['Asymmetry_1_3']))/3.0) #columns = ['Asymmetry_1'])
    a_2 = ((minmax_scale(asym2['Asymmetry_2_1'])+ minmax_scale(asym2['Asymmetry_2_2']) + minmax_scale(asym2['Asymmetry_2_3']))/3.0) #columns = ['Asymmetry_2'])
    a_3 = ((minmax_scale(asym3['Asymmetry_3_1'])+ minmax_scale(asym3['Asymmetry_3_2']) + minmax_scale(asym3['Asymmetry_3_3']))/3.0) #columns = ['Asymmetry_3'])
    a_4 = ((minmax_scale(asym4['Asymmetry_4_1'])+ minmax_scale(asym4['Asymmetry_4_3']) + minmax_scale(asym4['Asymmetry_4_5']))/3.0) #columns = ['Asymmetry_4'])
    a_5 = ((minmax_scale(asym5['Asymmetry_5_1'])+ minmax_scale(asym5['Asymmetry_5_2']) + minmax_scale(asym5['Asymmetry_5_3']))/3.0) #columns = ['Asymmetry_5'])
    a_6 = ((minmax_scale(asym6['Asymmetry_6_1'])+ minmax_scale(asym6['Asymmetry_6_2']) + minmax_scale(asym6['Asymmetry_6_3']))/3.0) #columns = ['Asymmetry_6'])
    a_7 = ((minmax_scale(asym7['Asymmetry_7_1'])+ minmax_scale(asym7['Asymmetry_7_2']) + minmax_scale(asym7['Asymmetry_7_3'])+ 
            minmax_scale(asym7['Asymmetry_7_4']) + minmax_scale(asym7['Asymmetry_7_5']) + minmax_scale(asym7['Asymmetry_7_6']))/6.0) #columns = ['Asymmetry_7'])
    a_8 = ((minmax_scale(asym8['Asymmetry_8_1'])+ minmax_scale(asym8['Asymmetry_8_2']) +minmax_scale(asym8['Asymmetry_8_3']))/3.0) #columns = ['Asymmetry_8'])
    
    #combine 
    asymm_label=np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8))
    asym_label = df(asymm_label, columns = ['Annotations'])
    asym_label = asym_label.reset_index()
    annotations = asym_label.iloc[:,1]
    
    #groundtruth annotations 
    truth =  pd.concat([data1['Melanoma'], data2['Melanoma'], data3['Melanoma'], data4['Melanoma'], data5['Melanoma'], data6['Melanoma'],data7['Melanoma'], data8['Melanoma']],axis =0, sort = False)
    truth = truth.reset_index()
    truth = truth.iloc[:,1]

    frames = [annotations, truth]
    result = pd.concat(frames, axis =1, sort = False)

    #raincloud plot annotaions 
    import seaborn as sns 
    import ptitprince as pt 
    data = result 
    sns.set(style = 'white', font_scale = 1.5)
    dx='Melanoma'; dy ='Annotations' 
    ort ='v' 
    pal ='Set2'
    sigma =.2
    ax = pt.RainCloud(x =dx, y =dy , data = data, palette = 'Set2', width_viol =.4, width_box =.2, figsize =(7,5), orient= 'h', move=.3)
    

    
    