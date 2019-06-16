# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 12:39:50 2019

@author: s162152
"""

from classify_automatic import *
from classify_visual import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathtofiles import *

mask="given" #or "created"
option="visual" #or '2000' or "2000all"


##Load the data of the created or given masks
if mask=="created":
    df1=pd.read_csv(pathtofiles()+'csvfiles/output_created_masks.csv')
elif mask=="given":
    df1=pd.read_csv(pathtofiles()+'csvfiles/output.csv')

##Gather the classification data
scores_LR_1,scores_svm_1, scores_LR_2,scores_svm_2,scores_LR_3,scores_svm_3,scores_LR_combined,scores_svm_combined=classify_automatic(df1,option,mask)


##For each option the AUC scores are collected in a dictionary
if option=="visual":
    scores_svm_vis, scores_LR_vis=classify_visual()
    data={"scores_LR_comp":scores_LR_1,"scores_svm_comp":scores_svm_1,"scores_LR_conv":scores_LR_2,"scores_svm_conv":scores_svm_2,"scores_LR_abr":scores_LR_3,"scores_svm_abr":scores_svm_3,"scores_LR_combined":scores_LR_combined,"scores_svm_combined":scores_svm_combined,"scores_LR_vis":scores_LR_vis,"scores_svm_vis":scores_svm_vis}
elif option=="2000":
      data={"scores_LR_comp":scores_LR_1,"scores_svm_comp":scores_svm_1,"scores_LR_conv":scores_LR_2,"scores_svm_conv":scores_svm_2,"scores_LR_abr":scores_LR_3,"scores_svm_abr":scores_svm_3,"scores_LR_combined":scores_LR_combined,"scores_svm_combined":scores_svm_combined,"scores_LR_vis":scores_LR_vis,"scores_svm_vis":scores_svm_vis}
elif option=="2000all":
      data={"scores_LR_border":scores_LR_1,"scores_svm_border":scores_svm_1,"scores_LR_asymmetry":scores_LR_2,"scores_svm_asymmetry":scores_svm_2,"scores_LR_color":scores_LR_3,"scores_svm_color":scores_svm_3,"scores_LR_combined":scores_LR_combined,"scores_svm_combined":scores_svm_combined}

    

## Create a figure instance
fig = plt.figure(1, figsize=(9, 6))

## Create an axes instance
ax = fig.add_subplot(111)

##Draw boxplots of AUC scores
bp = ax.boxplot(data.values())
ax.set_xticklabels(data.keys(),rotation="vertical")
    
plt.grid(None)
ax.set_ylabel('AUC score')
    