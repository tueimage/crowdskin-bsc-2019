# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:28:04 2019

@author: s162152
"""

import csv
import sys
import os

import matplotlib.pyplot as plt
from pathtofiles import *

import cv2
import numpy as np
import pandas as pd



    # you can provide both the input and output directory
    # yourself, by default dir_in will be the current dir
    # and dir_out will be <current_dir>/out/
def main():
    old=[]
    ##change this to the path where the images are saved
    dir_input=pathtofiles()+'image_data'
    print(dir_input)

    
    for f in os.listdir(dir_input):
        if os.path.isfile(f) and f[-4:] == ".jpg":
            y=str(f)
            img = cv2.imread(f,0)


## preprocess image            
            if img is not None:
                blur = cv2.GaussianBlur(img,(5,5),0)
                ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
          
                kernel=np.ones((9,9),np.uint8)
                th3=cv2.dilate(th3, kernel,iterations=1)
                th3=255-th3
##determine contours in image               
                im2,contours,hierarchy = cv2.findContours(th3,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                mask = np.ones(im2.shape[:2], dtype="uint8") * 255

##find the biggest contour area==lesion area               
                maxContourArea = 0;
                for c in contours:
                	area = cv2.contourArea(c)
                	if area > maxContourArea:
                		maxContourArea = area

                for c in contours:
## if the contour is bad, draw it on the mask
                		area = cv2.contourArea(c)
                       

                		if area < maxContourArea:
                			cv2.drawContours(mask, [c], -1, 0, -1)
                
##Extract the lesion area and determine if the lesion area touches the border of the image                
                image = cv2.bitwise_and(im2, im2, mask=mask)
          
                        
                im6,contours,hierarchy = cv2.findContours(image, 1, 2)
                contourLen = len(contours)
                cnt = contours[contourLen-1]

                xmin=0
                ymin=0
                xmax=img.shape[1]-1
                ymax=img.shape[0]-1
                yes=[]
                
                for i in range(ymax):
                    for j in range(xmax):
                        dist1 = cv2.pointPolygonTest(cnt,(xmin,i),False)
                        dist2 = cv2.pointPolygonTest(cnt,(xmax,i),False)
                        dist3 = cv2.pointPolygonTest(cnt,(j,ymin),False)
                        dist4 = cv2.pointPolygonTest(cnt,(j,ymax),False)
                        #print(dist1)
                        if dist1==0.0 or dist2==0.0 or dist3==0.0 or dist4==0.0:
                            yes.append(1.0)
                            break
                    else:
                        continue
                    break
##append the images from which the lesion area touches the border of the image to a csv file                
                if 1.0 in yes:
                    old.append(y)

    with open(pathtofiles()+'csvfiles/use_given_masks.csv', 'w',newline='') as csvfile:
        fieldnames = ['image']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in old:
            writer.writerow({'image': i})

if __name__ == "__main__": main()