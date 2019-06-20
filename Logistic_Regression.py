# -*- coding: utf-8 -*-
"""
Logistic regression function using ski-kitlearn.
Also SVC is included later but not used.

@author: s151385
"""

def loadData(filename):
    import pandas as pd
    import numpy as np
    import os
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler
    from sklearn import preprocessing
    from config import LESION_PATH as LESION_PATH
    from config import MASK_PATH as MASK_PATH
    from config import MAIN_PATH as MAIN_PATH
    os.chdir(MAIN_PATH)
    data_raw = pd.read_csv(filename)
    classes = []
    for i in range(1,10):
         classes.append('p'+str(i))

    x_data_raw = data_raw[classes]
    y_data = data_raw['melanoom_truth']
    x_data = preprocessing.scale(x_data_raw)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33, random_state=None, shuffle = True)
    return(x_train,y_train,x_test,y_test)
    
def logReg():
    # logistic regression
    from CM_ROC import CM, ROC
    x_train,y_train,x_test,y_test = loadData('data_final_variegation.csv')
    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression(max_iter = 1000000,dual=False, class_weight='balanced',
                                      solver='liblinear')
    logisticRegr.fit(x_train, y_train)
    predictions = logisticRegr.predict(x_test)
    score = logisticRegr.score(x_test,y_test)
    print(score)
    #CM(y_test,predictions)
    roc_auc = ROC(y_test,predictions)
    return(roc_auc)
        
def SVC():
    # support vector classification with stochasitic gradient descent
    from CM_ROC import CM, ROC
    x_train,y_train,x_test,y_test = loadData('data_final_variegation.csv')
    from sklearn import linear_model
    clf = linear_model.SGDClassifier(loss = 'hinge',max_iter = 10000, class_weight = 'balanced')
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    score = clf.score(x_test,y_test)
    print(score)
    CM(y_test,predictions)
    roc_auc = ROC(y_test,predictions)
    return(roc_auc)
    

roc_new = []
for i in range(0,100):
    print(i)
    roc_new.append(logReg())
    roc_final_logReg = np.array(roc_new)
    
import matplotlib.pyplot as plt
plt.boxplot(roc_new)