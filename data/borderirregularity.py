# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:05:12 2019

@author: s162152
"""

from binarize import *
import skimage
from skimage.measure import perimeter, label, regionprops
from skimage import color
import math
from centroid import *
import cv2
import sys
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt



def compactness(image):
    im6,contours,hierarchy = cv2.findContours(image, 1, 2)
    im6=255-im6
    im6 = binarize(im6)
                       
    ##Find area and perimeter    
    label_img = label(im6)
    region = regionprops(label_img)
                      
    img_area = max([props.area for props in region]) #Want the max because they could be many spaces
    img_perimeter = max([props.perimeter for props in region])
                        
    ##Calculate CI's formula
    CI=((img_perimeter**2) / (4.*math.pi*img_area))
    CI=np.asarray(CI)

        
    return(CI)
    
def convexity(image):
    im6,contours,hierarchy = cv2.findContours(image, 1, 2)
    contourLen = len(contours)
    cnt = contours[contourLen-1]
    lesionArea = cv2.contourArea(cnt)
    
    ##Make the convex hull and find the deviations with the lesion area
    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)
              
    if defects is not None:
        farSum = 0.0
        count = 0
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            cv2.line(im6,start,end,[0,255,0],2)
            ##sum the distances to farthest point
            farSum = farSum + d/256.0
            count = count + 1
        im6 = 255 - im6
        farAverage = farSum / count
        convex = farAverage/lesionArea
    
        return convex
    
def abruptness(image):    
    im6,contours,hierarchy = cv2.findContours(image, 1, 2)
    im6=255-im6

    
    ##Get the centroid and distances to edges
    _max_rad_pt, _max_rad, outline_img, centroid_to_edges, centroid = get_centroid(curr_img=im6)
    ##Get the average distance
    mean_dist = np.mean(centroid_to_edges)
    binarized_img = im6.astype(bool)
    binarized_img = ndimage.binary_fill_holes(binarized_img)
                
    ##Get the perimeter of image
    label_img = label(binarized_img)
    region = regionprops(label_img)
    img_perimeter = max([props.perimeter for props in region])
    edge_score = 0.
    for d in centroid_to_edges:
        edge_score += (d - mean_dist)**2
               
    edge_score /= (img_perimeter * (mean_dist**2))
                
    edge_score=np.asarray(edge_score)
    return(edge_score)
