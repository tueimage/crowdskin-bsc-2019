# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:07:46 2019

@author: s151385
"""

def loadData(filename):
    import csv
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import normalize
    x_data = np.zeros([2000,9])
    y_data = np.zeros([2000])
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset = list(lines)
        dataset = dataset[1:2002]
        for x in range(0,len(dataset)-1):
            for y in range(1,10):
                x_data[x,y-1] = float(dataset[x][y])
            y_data[x] = float(dataset[x][10])
        #normalize dataset on x_data so x_test and x_train same normalization
        x_data = normalize(x_data,axis=1)
        x_train = x_data[600:2000,:]
        x_test = x_data[0:600,:]
        y_train = y_data[600:2000]
        y_test = y_data[0:600]
        return(x_train,y_train,x_test,y_test)

def loadData2(filename):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler
    data_raw = pd.read_csv(filename)
    classes = []
    for i in range(1,10):
         classes.append('p'+str(i))

    x_data_raw = data_raw[classes]
    #y_data = data_raw['melanoom_truth']
    x_data = normalize(x_data_raw,axis=0)
# =============================================================================
#     k = np.load('k_sse_2000.npy')
#     x_data = normalize(k,axis=1)
#     x_data = normalize(x_data,axis=0)
# =============================================================================
    melanoom_truth = np.load('melanoom_truth.npy')
    y_data = melanoom_truth[0:2000]
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33, random_state=None, shuffle = True,stratify=y_data)
    return(x_train,y_train,x_test,y_test)
    
def loadData_k(filename):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    from sklearn.preprocessing import MinMaxScaler
    
    melanoom_truth = np.load('melanoom_truth.npy')
    x_data = normalize(k_new,axis=0)
    y_data = melanoom_truth[0:2000]
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.33, random_state=None, shuffle = True)
    return(x_train,y_train,x_test,y_test)

def kNN():
    #kNN
    from CM_ROC import CM, ROC
    x_train,y_train,x_test,y_test = loadData2('data_color.csv')
    m = sum(y_train)
    l = len(y_train)
    index = []
    for i in range(0,l):
        if y_train[i] == '1.0':
            index.append[i]
    from sklearn.neighbors import KNeighborsClassifier
    # n = 45, because sqrt(2000)
    neigh = KNeighborsClassifier(n_neighbors=45)
    neigh.fit(x_train,y_train)
    predictions = neigh.predict(x_test)
    roc_auc = ROC(y_test,predictions)
    return(roc_auc)
    
def logReg():
    # logistic regression
    from CM_ROC import CM, ROC
    x_train,y_train,x_test,y_test = loadData2('data.csv')
    #correction for more benign in trainingset
    p = sum(y_train)/len(y_train)
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
    x_train,y_train,x_test,y_test = loadData2('data_color.csv')
    p = sum(y_train)/len(y_train)
    from sklearn import linear_model
    clf = linear_model.SGDClassifier(loss = 'hinge',max_iter = 10000, class_weight = 'balanced')#{0.0: p, 1.0: 1})
    clf.fit(x_train,y_train)
    predictions = clf.predict(x_test)
    score = clf.score(x_test,y_test)
    print(score)
    CM(y_test,predictions)
    roc_auc = ROC(y_test,predictions)
    return(roc_auc)
    

roc_new = []
for i in range(0,100):
    roc_new.append(logReg())
