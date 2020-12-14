# -*- coding: utf-8 -*-
"""
Principal Component Analysis performend on ABC crowd annotations.

@author: s151385
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import pandas as pd
from Crowd_LogReg import loadData
data_all = loadData()
X = data_all[:,0:3]
pca = PCA(n_components=2) 
principalComponents = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
#melanoom_truth = data_raw['melanoom_truth']
true = data_all[:,3]
melanoom_truth = pd.DataFrame(data_all[:,3])
finalDf = pd.concat([principalDf, melanoom_truth[0:2000]], axis = 1)


k=0
j=0
pc_mel = np.zeros([595,2])
pc_ben = np.zeros([592,2])
for i in range(0,592):
    if true[i] == 1.0:
        pc_mel[j,:] = finalDf.iloc[i,0:2]
        j = j+1
    else:
        pc_ben[k,:] = finalDf.iloc[i,0:2]
        k = k+1

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.scatter(pc_mel[:,0], pc_mel[:,1], s=10, c='b', marker="x", label='Melanoma')
ax1.scatter(pc_ben[:,0],pc_ben[:,1], s=10, c='r', marker="x", label='Benign')
plt.legend(loc='upper left');
plt.show()
