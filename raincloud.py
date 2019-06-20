# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:45:53 2019
Rain cloud plots
@author: s151385
"""

import os
from config import LESION_PATH as LESION_PATH
from config import MASK_PATH as MASK_PATH
from config import MAIN_PATH as MAIN_PATH
os.chdir('MAIN_PATH')
def RainCloud():
    import pandas as pd 
    import seaborn as sns
    import ptitprince as pt
    data = pd.read_csv('data_final_Emiel.csv', header =0)
    from sklearn.preprocessing import normalize
    #method 1 
    sns.set(style = 'white', font_scale = 1.5)
    dx = 'melanoom_truth'
    dy = 'p8'
    ort ='v' 
    pal ='Set2'
    sigma =.2
    ax = pt.RainCloud(x =dx, y =dy , data = data, width_viol =.4, width_box =.2, figsize =(7,5), orient= 'h', move=.3)
    ax.set_ylabel('')
    ax.set_xlabel('Melanoma',weight='bold')
    plt.savefig('namen.png')
RainCloud()


