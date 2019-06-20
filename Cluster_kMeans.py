# -*- coding: utf-8 -*-
"""
k-means clustering to determine SSD for first 5 k's.
Afterwards the correlation coefficient will be calculated and used as feature
for classification.
Returns csv file with image name, 4x rc and ground truth

@author: s151385
"""

import os
import cv2
from config import LESION_PATH as LESION_PATH
from config import MASK_PATH as MASK_PATH
from config import MAIN_PATH as MAIN_PATH
    #%%
def cluster(img_name):
    img_name = "ISIC_0000000"
    os.chdir(MASK_PATH);
    img_name_mask = img_name+'_segmentation.png'
    img_mask = cv2.imread(img_name_mask)
    
    #%%
    mask = img_mask[:,:,0]/255
    
    #%%
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    os.chdir(LESION_PATH);
    img = cv2.imread(img_name+'.jpg',0)
    
    width = 500
    wpercent = (width / float(len(img[0,:])))
    height = int((float(len(img[:,0])) * float(wpercent)))
    
    img = cv2.resize(img,(width,height))
    mask = cv2.resize(mask,(width,height))
    
    img = img*mask
    
    mms = MinMaxScaler()
    mms.fit(img)
    img_transformed = mms.transform(img)
    
    Sum_of_squared_distances = []
    K = range(1,6)
    for k in K:
        km = KMeans(n_clusters=k)
        km = km.fit(img_transformed)
        Sum_of_squared_distances.append(km.inertia_)

# =============================================================================
#     print(Sum_of_squared_distances)    
#     plt.plot(K,Sum_of_squared_distances,'bx-')
#     plt.xlabel('k')
#     plt.ylabel('Sum_of_Squared_Distances')
#     plt.title('Elbow Method to determine Optimal k')
#     plt.show()
# =============================================================================
    return(Sum_of_squared_distances[0],Sum_of_squared_distances[1],Sum_of_squared_distances[2],Sum_of_squared_distances[3],Sum_of_squared_distances[4])

#%%open ground truth file 
import csv
from sklearn.preprocessing import normalize
melanoom_truth = []
image_names = [] 
import numpy as np
os.chdir(MAIN_PATH)
with open('ISIC-2017_Training_Part3_GroundTruth.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader,None) #skip headers 
    for lines in reader:
        image_names.append(str(lines[0]))
        melanoom_truth.append(str(lines[1]))           

ssd = np.zeros([2000,5])
for i in range(0,2000):
    image_name = image_names[i]
    ssd[i,:] = cluster(image_name)
    print(i)
    
k = normalize(ssd,axis=1)
cc= np.zeros([len(k),4])
for i in range(0,len(k)):
    for j in range(0,4):
        cc[i,j] = k[i,j+1]-k[i,j]
        
#%% save results in csv file 
os.chdir('/Users/s151385/OneDrive - TU Eindhoven/1. BEP MIA/Color/')
with open('data_cluster.csv', 'w', newline='') as f:
    fieldnames = [ 'imageID', 'cc1','cc2','cc3','cc4', 'melanoom_truth']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for p in range(len(cc)):
        writer.writerow({'imageID': image_names[p], 'cc1': cc[p,0],'cc2': cc[p,1],
                         'cc3': cc[p,2],'cc4': cc[p,3],'melanoom_truth': melanoom_truth[p]})

