# -*- coding: utf-8 -*-
"""
Logistic Regression on my own automated features and that of Sanne and Audrey.
Return numpy array of AUC values

@author: s151385
"""

def loadData_Emiel(filename):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    data_raw = pd.read_csv(filename)
    classes = ['p1', 'p4','p5','p6','p7','p8']
    x_data_raw = data_raw[classes]
    y_data = data_raw['melanoom_truth']
    x_data_Emiel = normalize(x_data_raw,axis=0)
    #x_data_Emiel = rc
    return(x_data_Emiel, y_data)
    
def loadData_Sanne(filename):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    data_raw = pd.read_csv(filename)
    classes = ['asymmetry']
    x_data_raw = data_raw[classes]
    x_data_Sanne = normalize(x_data_raw,axis=0)
    return(x_data_Sanne)
    
def loadData_Audrey(filename):
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import normalize
    data_raw = pd.read_csv(filename)
    classes = ['compactness','abruptness']
    x_data_raw = data_raw[classes]
    x_data_Audrey = x_data_raw#normalize(x_data_raw,axis=0)
    return(x_data_Audrey)

def main():   
    x_data_Audrey = loadData_Audrey('data_audrey.csv')
    x_data_Emiel,y_data = loadData_Emiel('data_final_variegation.csv')
    x_data_Sanne = loadData_Sanne('data_Sanne.csv')
    x_data_Sanne_2 = loadData_Sanne('data_Sanne_2.csv')
    
    x_final = np.concatenate((x_data_Audrey,x_data_Emiel,x_data_Sanne,x_data_Sanne_2),axis=1)
    
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x_final, y_data, test_size=0.33, random_state=None, shuffle = True)
    
    from CM_ROC import CM, ROC
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
    
roc = []
for i in range(0,10):
    print(i)
    roc.append(main())
    roc_all_new = np.array(roc)
plt.boxplot(roc)
