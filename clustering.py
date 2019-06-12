from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
import os
import cv2
    #%%
def cluster(img_name):
    #img_name = "ISIC_0000013"
    os.chdir('/Users/s151385/OneDrive - TU Eindhoven/1. BEP MIA/Color/Mask/');
    img_name_mask = img_name+'_segmentation.png'
    img_mask = cv2.imread(img_name_mask)
    
    #%%
    mask = img_mask[:,:,0]/255
    
    #%%
    import time
    start_time = time.time()
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from sklearn.metrics import silhouette_score
    os.chdir('/Users/s151385/OneDrive - TU Eindhoven/1. BEP MIA/Color/Lesion/');
    img = cv2.imread(img_name+'.jpg',0)
    
    width = 500
    wpercent = (width / float(len(img[0,:])))
    height = int((float(len(img[:,0])) * float(wpercent)))
    
    img = cv2.resize(img,(width,height))
    mask = cv2.resize(mask,(width,height))
    
    img = img*mask
    
    
# =============================================================================
#     mms = MinMaxScaler()
#     mms.fit(img)
#     img_transformed = mms.transform(img)
# =============================================================================
    
# =============================================================================
#     Sum_of_squared_distances = []
#     K = range(1,6)
#     for k in K:
#         km = KMeans(n_clusters=k)
#         km = km.fit(img_transformed)
#         Sum_of_squared_distances.append(km.inertia_)
# =============================================================================
    img_transformed = img    
    sil = []
    l = 0
    for n_cluster in range(2, 20,5):
        kmeans = KMeans(n_clusters=n_cluster).fit(img_transformed)
        label = kmeans.labels_
        sil_coeff = silhouette_score(img_transformed, label, metric='euclidean')
        sil.append(sil_coeff)
        l = l+1
        #print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))

# =============================================================================
#     af = AffinityPropagation().fit(img)
#     cluster_centers_indices = af.cluster_centers_indices_
#     n_clusters_ = len(cluster_centers_indices)
#     
#     af_color = AffinityPropagation().fit(img_r)
#     cluster_centers_indices_color = af_color.cluster_centers_indices_
#     n_clusters__color = len(cluster_centers_indices_color)
# =============================================================================

        
# =============================================================================
#     plt.plot(Sum_of_squared_distances, 'bx-')
#     plt.xlabel('k')
#     plt.ylabel('Sum_of_squared_distances')
#     plt.title('Elbow Method For Optimal k')
#     plt.show()
# =============================================================================
    #print("--- %s seconds ---" % (time.time() - start_time))
    return(sil)

#%%open ground truth file 
import os
import csv
melanoom_truth = []
image_names = [] 
import numpy as np
os.chdir('/Users/s151385/OneDrive - TU Eindhoven/1. BEP MIA/Color/')
with open('ISIC-2017_Training_Part3_GroundTruth.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader,None) #skip headers 
    for lines in reader:
        image_names.append(str(lines[0]))
        melanoom_truth.append(str(lines[1]))           

k_new = np.zeros([2000,4])
for i in range(0,2000):
    image_name = image_names[i]
    k_new[i,:] = cluster(image_name)
    print(i)
    