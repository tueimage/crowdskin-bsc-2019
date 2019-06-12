# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 09:37:38 2019

@author: s151385
"""
# =============================================================================
# 
# import matplotlib.pyplot as plt
# roc_plot = [roc_logReg_balanced, roc_logReg_p, roc_logReg_cg, roc_SVC]
# label = ['balanced', 'p', 'newton-cg', 'SVC']
# 
# fig = plt.figure()
# fig.suptitle('Boxplots autodetected Color', fontsize=14, fontweight='bold')
# 
# ax = fig.add_subplot(111)
# ax.boxplot(roc_plot,labels=label)
# 
# ax.set_title('Logistic Regression, (n=50)')
# ax.set_ylabel('AUC')
# plt.savefig('boxplots_color.png')
# =============================================================================

import matplotlib.pyplot as plt
# =============================================================================
# roc_plot = [roc_color,roc_all, roc_cluster]
# label = ['Color','All', 'Cluster Elbow']
# =============================================================================

roc_plot = [roc_01_color,roc_06_color, roc_07_color]
label = ['01','06', '07']

fig = plt.figure()
fig.suptitle('AUC crowd sourced Color', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
ax.boxplot(roc_plot,labels=label)

ax.set_title('Logistic Regression')
ax.set_ylabel('AUC')
ax.set_xlabel('Group')
plt.savefig('boxplots_crowd_new.png')

