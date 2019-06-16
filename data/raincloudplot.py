# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 09:17:52 2019

@author: s162152
"""

"""
Created on Wed Jun  5 12:29:12 2019

@author: s162152
"""
import pandas as pd
import ptitprince as pt
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="whitegrid",font_scale=2)
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from pathtofiles import *

###Import data of different groups
df_1 = pd.read_excel(pathtofiles()+'groups/group01.xlsx')
df_2 = pd.read_excel(pathtofiles()+'groups/group02.xlsx')
df_3 = pd.read_excel(pathtofiles()+'groups/group03.xlsx')
df_4 = pd.read_excel(pathtofiles()+'groups/group04.xlsx')
df_5 = pd.read_excel(pathtofiles()+'groups/group05.xlsx')
df_6 = pd.read_excel(pathtofiles()+'groups/group06.xlsx')
df_7 = pd.read_excel(pathtofiles()+'groups/group07.xlsx')
df_3 = df_3.reset_index()
df_7 = df_7.dropna()
df_7 = df_7.reset_index()

##Scale and average the visual scores for border to be able to merge them
a_1 = (preprocessing.scale(df_1['Border_1_1']) + preprocessing.scale(df_1['Border_1_2']) + preprocessing.scale(df_1['Border_1_3'])) / 3.0
a_2 = (preprocessing.scale(df_2['Rand onregelmatigheid']) + preprocessing.scale(df_2['Unnamed: 6']) + preprocessing.scale(df_2['Unnamed: 7'])) / 3.0
a_3 = (preprocessing.scale(df_3['Border_3_1']) + preprocessing.scale(df_3['Border_3_2']) + preprocessing.scale(df_3['Border_3_3'])) / 3.0
a_4 = (preprocessing.scale(df_4['Border_4_1']) + preprocessing.scale(df_4['Border_4_3']) + preprocessing.scale(df_4['Border_4_5'])) / 3.0
a_5 = (preprocessing.scale(df_5['Border_5_1']) + preprocessing.scale(df_5['Border_5_2']) + preprocessing.scale(df_5['Border_5_3'])) / 3.0
a_6 = (preprocessing.scale(df_6['Border_6_1']) + preprocessing.scale(df_6['Border_6_2']) + preprocessing.scale(df_6['Border_6_3'])) / 3.0
a_7 = (preprocessing.scale(df_7['Border_7_1']) + preprocessing.scale(df_7['Border_7_2']) + preprocessing.scale(df_7['Border_7_3']) + preprocessing.scale(df_7['Border_7_4']) + preprocessing.scale(df_7['Border_7_5']) + preprocessing.scale(df_7['Border_7_6'])) / 6.0

##merge the visual scores for border and reshape the labels (-1,1), because it is one feature
group_label=np.concatenate((a_1, a_2, a_3, a_4, a_5, a_6, a_7))
group_label=group_label.reshape(-1,1)
group_id=np.concatenate((df_1['ID'],df_2['Afbeelding'],df_3['Unnamed: 0'],df_4['ID'],df_5['ID'],df_6['ID'],df_7['ID'])) 

##load the data and delete the rows which contain 'None' or 'inf'
df1=pd.read_csv(pathtofiles()+'csvfiles/output_created_masks.csv')
##For the created masks: df1=pd.read_csv("output_created_masks.csv")
df1=df1.mask(df1.eq('None')).dropna()
df1=df1.mask(df1.eq('inf')).dropna()

##Load the groundtruth
df = pd.read_csv(pathtofiles()+'csvfiles/GroundTruth.csv')
class_label=df['melanoma']
class_id=df['image_id']

##Add the groundtruth to the dataset
df1['true']=class_label

##Find the labels and IDs from the groups in the groundtruth 
true_label=[]
true_id=[]

for i in range(len(group_id)):
    for j in range(len(class_id)):
        if group_id[i]==class_id[j]:
            true_label.append(class_label[j])
            true_id.append(class_id[j])
            


##Keep only the data of the images with a visual score
for i in range(len(true_id)):
    true_id[i]=str(true_id[i])+str('.jpg')

df1=df1[df1['image'].isin(true_id)]



border_id=df1['image']
border_label=df1.loc[:,['compactness','convex','abruptness']]
border_label=border_label.reset_index(drop=True)

##Scale the dataset with mean unit variance
scaler = StandardScaler()
border_label=scaler.fit_transform(border_label)
group_label=scaler.fit_transform(group_label)


## To make the rain cloud plot the different methods have to be organised below each other and divided into different groups
compactness=border_label[:,0]
convex=border_label[:,1]
abruptness=border_label[:,2]
groundtruth=df1['true']
groundtruth=groundtruth.reset_index(drop=True)


compactness=pd.DataFrame(compactness)
convex=pd.DataFrame(convex)
abruptness=pd.DataFrame(abruptness)
visual=pd.DataFrame(group_label)

compactness=pd.concat([compactness,pd.DataFrame({'colourgroup':'1.0'},index=compactness.index),pd.DataFrame({'true':groundtruth},index=compactness.index)],axis=1)
convex=pd.concat([convex,pd.DataFrame({'colourgroup':'2.0'}, index=convex.index),pd.DataFrame({'true':groundtruth},index=convex.index)],axis=1)
abruptness=pd.concat([abruptness,pd.DataFrame({'colourgroup':'3.0'}, index=abruptness.index),pd.DataFrame({'true':groundtruth},index=abruptness.index)],axis=1)
visual=pd.concat([visual,pd.DataFrame({'colourgroup':'4.0'}, index=visual.index),pd.DataFrame({'true':true_label},index=visual.index)],axis=1)



onedf=pd.concat([compactness,convex,abruptness,visual])
onedf=np.array(onedf,np.float64)
true_label=np.array(true_label,np.float64)
border_label=np.array(border_label,np.float64)
group_label=np.array(group_label,np.float64)



##Make the rain cloud plot
f, ax = plt.subplots(figsize=(20,10))
dy=onedf[:,1];dx=onedf[:,0]; ort="h";dhue =onedf[:,2]; pal = "Set2"
ax=pt.half_violinplot( x = dx, y = dy, hue=dhue,split=True, scale_hue=False, palette = pal, bw=.2, cut = 0.,scale = "area",width=.6, inner = None, orient = ort)
ax=sns.stripplot( x = dx, y = dy,hue=dhue, palette = pal, edgecolor = "white",
                 size = 3, jitter = 1, zorder = 0, orient = ort)
ax=sns.boxplot( x = dx, y = dy,hue=dhue, palette=pal,dodge=True, width = .1, zorder = 10,\
            showcaps = True, boxprops = { "zorder":10},\
            showfliers=True, whiskerprops = {'linewidth':2, "zorder":10},\
               saturation = 1, orient = ort)
ax.set_yticklabels(['compactness','convexity','abruptness', 'visual'],rotation="horizontal")
plt.title("Raincloud with Boxplot")
