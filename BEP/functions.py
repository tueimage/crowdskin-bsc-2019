# -*- coding: utf-8 -*-
"""
Created on Wed May 22 09:14:00 2019

@author: s163729
"""

def shapiro(data):
    import pandas as pd 
    from scipy import stats
    #read data set 
    data = pd.read_csv(data, header =0)
    asym1 = data['asymmetry1']
    asym2 = data['asymmetry2']
    #apply shapiro-wilk test     
    return stats.shapiro(asym1)
    return stats.shapiro(asym2)
    
def ztest(data):
    import pandas as pd 
    from statsmodels.stats import weightstats as stests 
    #read data set 
    data = pd.read_csv(data, header =0)
    asym1= data['asymmetry1']
    asym2 = data['asymmetry2']
    
    #apply z-test
    ztest, pval = stests.ztest(asym1,x2 = asym2, value = 0, alternative ='two-sided')
    print(float(pval))
    if pval < 0.5:
        print('reject null hypothesis')
    else:
        print('accept null hypothesis')

    
def meanANDstd(data):
    #determine mean and standard deviation of asymmetry 
    import pandas as pd 
    import numpy as np 
    import matplotlib.pyplot as plt 
    
    #read in csv file 
    data = pd.read_csv(data,header = 0)
    data = data.dropna()
    
    #split data frame 
    asymmetry = data['asymmetry1']
    asymmetry2 = data['asymmetry2']
    #determine the mean and std of all the data 
    mean_totaal = np.mean(asymmetry)
    mean2_totaal = np.mean(asymmetry2)
    std_totaal= np.std(asymmetry)
    std2_totaal= np.std(asymmetry2)
    
    return mean_totaal 
    return mean2_totaal 
    return std_totaal 
    return std2_totaal       
    

def RainCloud(data):
    import pandas as pd 
    import seaborn as sns
    import ptitprince as pt
    data = pd.read_csv(data, header =0)
    #method 1 
    sns.set(style = 'white', font_scale = 1.5)
    dx = 'melanoma_truth' ; dy = 'asymmetry1' 
    ort ='v' 
    pal ='Set2'
    sigma =.2
    ax = pt.RainCloud(x =dx, y =dy , data = data, palette = 'Set2', width_viol =.4, width_box =.2, figsize =(7,5), orient= 'h', move=.3)
    
    #method 2 
    sns.set(style = 'white', font_scale =1.5)
    dx2 = 'melanoma_truth'; dy2 = 'asymmetry2'
    ax2 = pt.RainCloud(x = dx2 , y = dy2, data = data, palette = 'Set2', width_viol=.4, width_box =.2, figsize = (7,5), orient = 'h',move=.3)
    


            

        