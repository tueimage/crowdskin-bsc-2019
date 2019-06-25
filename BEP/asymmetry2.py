# -*- coding: utf-8 -*-
"""
Created on Mon May 20 08:39:46 2019

@author: s163729
"""

def asymmetry2(maskedImage):
    
    import numpy as np 
    from skimage.measure import label, regionprops
    import cv2
    import imutils
    import matplotlib.pyplot as plt
    
    #read in mask 
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
    
    angle = angle*(180/3.14159)
    
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
 
    #rotate around its main axis and measure the asymmetry in both x and y  
    
    symY= []
    symX= []
    for i in np.arange(0,185,5):
        rotated = imutils.rotate_bound(rotatedImage,i)
        #plt.imshow(rotated)
        labeledImage2= label(rotated)
        regions2 = regionprops(labeledImage2)
        for props in regions2:
            y,x = props.centroid
        
        total_row2, total_col2 = labeledImage2.shape
        cen_x2, cen_y2 = total_col2/2, total_row2/2
        deltax2 = cen_x2 - x 
        deltay2 = cen_y2 - y 
        translation_matrix2= np.float32([[1,0,deltax2],[0,1,deltay2]])
        rotated= cv2.warpAffine(rotated,translation_matrix2, (total_col2,total_row2))
        amountPixels = rotated.sum()
        flippedy = np.fliplr(rotated)
        flippedx = np.flipud(rotated)
        overlappedy = flippedy & rotated
        overlappedx = flippedx & rotated 
        amountsymY = overlappedy.sum()
        amountsymX = overlappedx.sum()
        ratiosymY = amountsymY/amountPixels
        ratiosymX = amountsymX/amountPixels
        symY.append(float(ratiosymY))
        symX.append(float(ratiosymX))
    
    totalY = 0 
    for valy in symY:
        totalY += valy 
    asymmetryY = totalY/len(symY)
    
    totalX = 0 
    for  valx in symX:
        totalX += valx 
    asymmetryX = totalX/len(symX)
    
    #determine the asymmetry of the lesion
    asymmetry = ((asymmetryX + asymmetryY)/2)
    asymmetry = str(round(asymmetry,3))

    return asymmetry 


