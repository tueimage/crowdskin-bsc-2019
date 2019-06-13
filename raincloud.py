# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:45:53 2019
Rain cloud plots
@author: s151385
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
data_raw = pd.read_csv('data.csv')
col = 3
row = 3
sns.set(style="whitegrid")
fig, axes = plt.subplots(ncols=row*col, figsize=(10, 15))
ax = axes.ravel()
ax[0] = plt.subplot(row, col, 1)
ax[0] = sns.stripplot(x="melanoom_truth", y="p1", data=data_raw, jitter=True)
ax[0].set_ylabel('percentage %')
ax[0].set_xlabel('')
ax[0].set_title('red')

ax[1] = plt.subplot(row, col, 2)
ax[1] = sns.stripplot(x="melanoom_truth", y="p2", data=data_raw, jitter=True)
ax[1].set_ylabel('percentage %')
ax[1].set_xlabel('')
ax[1].set_title('green')

ax[2] = plt.subplot(row, col, 3, sharex=ax[0], sharey=ax[0])
ax[2] = sns.stripplot(x="melanoom_truth", y="p3", data=data_raw, jitter=True)
ax[2].set_ylabel('percentage %')
ax[2].set_xlabel('')
ax[2].set_title('blue')

ax[3] = plt.subplot(row, col, 4)
ax[3] = sns.stripplot(x="melanoom_truth", y="p4", data=data_raw, jitter=True)
ax[3].set_ylabel('percentage %')
ax[3].set_xlabel('')
ax[3].set_title('purple')

ax[4] = plt.subplot(row, col, 5)
ax[4] = sns.stripplot(x="melanoom_truth", y="p5", data=data_raw, jitter=True)
ax[4].set_ylabel('percentage %')
ax[4].set_title('red')
ax[4].set_ylabel('percentage %')
ax[4].set_xlabel('')
ax[4].set_title('brown')

ax[5] = plt.subplot(row, col, 6)
ax[5] = sns.stripplot(x="melanoom_truth", y="p6", data=data_raw, jitter=True)
ax[5].set_ylabel('percentage %')
ax[5].set_xlabel('')
ax[5].set_title('darkbrown')

ax[6] = plt.subplot(row, col, 7)
ax[6] = sns.stripplot(x="melanoom_truth", y="p7", data=data_raw, jitter=True)
ax[6].set_ylabel('percentage %')
ax[6].set_xlabel('')
ax[6].set_title('grey')

ax[7] = plt.subplot(row, col, 8)
ax[7] = sns.stripplot(x="melanoom_truth", y="p8", data=data_raw, jitter=True)
ax[7].set_ylabel('percentage %')
ax[7].set_xlabel('')
ax[7].set_title('black')

ax[8] = plt.subplot(row, col, 9)
ax[8] = sns.stripplot(x="melanoom_truth", y="p9", data=data_raw, jitter=True)
ax[8].set_ylabel('percentage %')
ax[8].set_xlabel('')
ax[8].set_title('white')

plt.savefig('raincloud.png')



