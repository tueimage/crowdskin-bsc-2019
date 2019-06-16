# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:21:11 2019

@author: s162152
"""
import cv2
import numpy as np

def preprocessing(img):
    ##Append a gaussian blur on the image and treshold the image afterwards
    ##dilate the image to decrease the impact of hairs in the image
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel=np.ones((9,9),np.uint8)
    th3=cv2.dilate(th3, kernel,iterations=1)
    th3=255-th3
    ## Find the contours in the image                   
    im2,contours,hierarchy = cv2.findContours(th3,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(im2.shape[:2], dtype="uint8") * 255
        
                 
                        
    maxContourArea = 0;
    ##Loop over the contours and find the biggest contour
    for c in contours:
        area = cv2.contourArea(c)
        if area > maxContourArea:
            maxContourArea = area
            
    ## loop over the contours
    for c in contours:
        ## if the contour is bad, draw it on the mask
        area = cv2.contourArea(c)
                               
        if area < maxContourArea:
            cv2.drawContours(mask, [c], -1, 0, -1)
                    
    ##append mask on image               
    image = cv2.bitwise_and(im2, im2, mask=mask)
    return image              