# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:32:02 2019

@author: s163729
"""

def asymmetry(maskedImage):
    import numpy as np 
    from skimage.measure import label, regionprops
    import cv2
    import imutils 
 
    mask = cv2.imread(maskedImage)
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    ret,binaryImage=cv2.threshold(gray_image,0,1,cv2.THRESH_BINARY)
    
    #label image 
    labeledImage=label(binaryImage)
    
    #determine the center of the lesion 
    regions = regionprops(labeledImage)
    for props in regions:
        y0,x0 = props.centroid
        angle = -props.orientation
    
    #determine the center of the total image 
    total_row,total_col= labeledImage.shape 
    cen_x,cen_y=total_col/2, total_row/2
  
    #move lesion to the middle of the image 
    deltax= cen_x-x0 
    deltay= cen_y-y0
    
    #translate the image 
    translation_matrix= np.float32([[1,0,deltax],[0,1,deltay]])
    binaryImage = cv2.warpAffine(binaryImage,translation_matrix, (total_col,total_row))

    #rotate the image 
    rotatedImage= imutils.rotate(binaryImage,angle)
    
    
    #determine the total amount of pixels in the lesion 
    amountPixels = rotatedImage.sum()
    
    #flip lesion over y-axis 
    flippedy= np.fliplr(rotatedImage)
    overlappedy = flippedy & rotatedImage
    amountsymY= overlappedy.sum()
    ratiosymY = amountsymY/amountPixels 

    #flip lesion over x-axis 
    flippedx = np.flipud(rotatedImage)
    overlappedx= flippedx & rotatedImage
    amountsymX = overlappedx.sum()
    ratiosymX = amountsymX/amountPixels 
    
    #determine the asymmetry of the lesion
    asymmetry = ((ratiosymX + ratiosymY)/2)
    asymmetry = str(round(asymmetry,3)) 
    return asymmetry 






  
    
    