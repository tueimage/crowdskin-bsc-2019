"""
color:
Script bepaald per plaatje hoeveel pixels een bepaalde kleur bevatten en
berekend hier later het percentage tov het gehele oppervlak van.
Eerst wordt gekeken welke RGB waarde het grootste is, daarna kleur bepaalt
doormiddel van if statements.

Hier staat ook de plot functie in geschreven.

@author: s151385
"""
def colorcoding(img_name,plots):
    ### imports 
    import pandas as pd
    import os
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt
    
    ### imread image 
    #img_name = "ISIC_0000052"
    os.chdir('/Users/s151385/OneDrive - TU Eindhoven/1. BEP MIA/Color/Mask/');
    img_name_mask = img_name+'_segmentation.png'
    img_mask = cv2.imread(img_name_mask)
    os.chdir('/Users/s151385/OneDrive - TU Eindhoven/1. BEP MIA/Color/Lesion/');
    img = cv2.imread(img_name+'.jpg')   #bgr!!!!!!!
    
    width = 500
    wpercent = (width / float(len(img[0,:])))
    height = int((float(len(img[:,0])) * float(wpercent)))
    
    img_b,img_g,img_r = cv2.split(img)
    img_b = cv2.resize(img_b,(width,height))
    img_r = cv2.resize(img_r,(width,height))
    img_g = cv2.resize(img_g,(width,height))
    
    ### oppervlak mask bepalen in pixels
    mask = img_mask[:,:,0]/255
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask,kernel,iterations = 10)
    mask = cv2.resize(mask,(width,height))
    A_mask =  sum(mask.ravel())
    
    ### alleen lesion in img, verdelen in rgb
    lesion_r = img_r*mask
    lesion_g = img_g*mask
    lesion_b = img_b*mask

    
    ### maten afbeelding
    x = len(lesion_r[:,:])
    y = len(lesion_r[1,:])
    
    ### lege kleurmaskers
    black = np.zeros([x,y],dtype=np.uint8)
    red = np.zeros([x,y],dtype=np.uint8)
    darkbrown = np.zeros([x,y],dtype=np.uint8)
    brown = np.zeros([x,y],dtype=np.uint8)
    grey = np.zeros([x,y],dtype=np.uint8)
    white = np.zeros([x,y],dtype=np.uint8)
    purple = np.zeros([x,y],dtype=np.uint8)
    blue = np.zeros([x,y],dtype=np.uint8)
    green = np.zeros([x,y],dtype=np.uint8)  #geel incl
    
    
    ### per pixel kleur bepalen en opslaan in masker
    for i in range(0,x-1):
        for j in range(0,y-1):
            r = lesion_r[i,j]
            g = lesion_g[i,j]
            b = lesion_b[i,j]
    ### black en white eerst, scheelt groot deel
            if lesion_r[i,j] < 50 and lesion_g[i,j] < 50 and lesion_b[i,j] < 50:
                black[i,j] = 1
            elif lesion_r[i,j] == 255 and lesion_g[i,j] == 255 and lesion_b[i,j] == 255:
                white[i,j] = 1
    ### rood hoogste waarde
            elif lesion_r[i,j] >= lesion_g[i,j] and lesion_r[i,j] >= lesion_b[i,j]:
                if r < 100:
                  darkbrown[i,j] = 1
                elif r < 150 :
                  if g < 50 and b < 100:
                      red[i,j] = 1
                  elif abs(g - b) < 20:
                      grey[i,j] = 1
                  elif g > b:
                      darkbrown[i,j] = 1
                  else:
                      purple[i,j] = 1
                elif r < 200 :
                  if g < 130 and b < 130:
                      red[i,j] = 1
                  elif g > b:
                      brown[i,j] = 1
                  elif abs(g - b) < 20:
                      grey[i,j] = 1
                  else:
                      purple[i,j] = 1
                elif r <= 255 :
                  if g < 150 and b < 150:
                      red[i,j] = 1
                  elif g > b:
                      brown[i,j] = 1
                  else:
                      purple[i,j] = 1
    ### groen hoogste waarde
            elif lesion_g[i,j] >= lesion_r[i,j] and lesion_g[i,j] >= lesion_b[i,j]:
                if g < 70:
                    black[i,j] = 1
                elif g < 100:
                    if r > 80 and b < 80:
                        darkbrown[i,j] = 1
                    elif b > 80:
                        grey[i,j] = 1
                    else: 
                        green[i,j] = 1
                elif g < 150:
                    if r > 130 and b < 120:
                        brown[i,j] = 1
                    elif r > 130 and b > 120:
                        grey[i,j] = 1
                    else: 
                        green[i,j] = 1
                elif g < 200:
                    if r > 180 and b < 150:
                        brown[i,j] = 1
                    elif r > 180 and b > 150:
                        grey[i,j] = 1
                    else: 
                        green[i,j] = 1
                elif g <= 255:
                    if r < 200 and b < 200:
                        green[i,j] = 1
                    else:
                        white[i,j] = 1
    ### blauw hoogste waarde
            elif lesion_b[i,j] >= lesion_r[i,j] and lesion_b[i,j] >= lesion_g[i,j]:
                if b < 100:
                    if r > 50 or g > 80:
                        if g > r:
                            grey[i,j] = 1
                        elif r >= g:
                            purple[i,j] = 1
                    else:
                        blue[i,j] = 1
                elif b < 150:
                    if r > 90:
                        if r >= g:
                            purple[i,j] = 1
                        elif g > r:
                            grey[i,j] = 1
                    else:
                        blue[i,j] = 1
                elif b < 200:
                    if r > 120:
                        if r >= g:
                            purple[i,j] = 1
                        elif g > r:
                            grey[i,j] = 1
                    else:
                        blue[i,j] = 1
                elif b <= 255:
                    if r > 130 and r > (g-40):
                        purple[i,j] = 1;
                    else: 
     
                        blue[i,j] = 1
    black = black*mask                
    ### oppervlak in pixels van kleuren
    A_red = sum(red.ravel())
    A_darkbrown = sum(darkbrown.ravel())
    A_brown = sum(brown.ravel())
    A_green = sum(green.ravel())
    A_blue = sum(blue.ravel())
    A_grey = sum(grey.ravel())
    A_white = sum(white.ravel())
    A_black = sum(black.ravel())
    A_purple = sum(purple.ravel())
    A_white = sum(white.ravel())
    
    ### percentage
    p_red = A_red/A_mask*100
    p_darkbrown = A_darkbrown/A_mask*100
    p_brown = A_brown/A_mask*100
    p_grey = A_grey/A_mask*100
    p_green = A_green/A_mask*100
    p_purple = A_purple/A_mask*100
    p_blue = A_blue/A_mask*100
    p_black = A_black/A_mask*100
    p_white = A_white/A_mask*100
    p = [p_red,p_green,p_blue,p_purple,p_brown,p_darkbrown,p_grey, p_black, p_white]
    
    if plots == 1:             
        ### plot settings
        col = 3
        row = 3
        fig, axes = plt.subplots(ncols=row*col, figsize=(10, 5))
        ax = axes.ravel()
        ax[0] = plt.subplot(row, col, 1)
        ax[1] = plt.subplot(row, col, 2)
        ax[2] = plt.subplot(row, col, 3, sharex=ax[0], sharey=ax[0])
        ax[3] = plt.subplot(row, col, 4)
        ax[4] = plt.subplot(row, col, 5)
        ax[5] = plt.subplot(row, col, 6)
        ax[6] = plt.subplot(row, col, 7)
        ax[7] = plt.subplot(row, col, 8)
        ax[8] = plt.subplot(row, col, 9)
        
        ### plot data
        ax[0].imshow(red)
        ax[0].set_title('red')
        ax[0].axis('off')
        ax[1].imshow(brown)
        ax[1].set_title('brown')
        ax[1].axis('off')
        ax[2].imshow(darkbrown)
        ax[2].set_title('darkbrown')
        ax[2].axis('off')
        ax[3].imshow(purple)
        ax[3].set_title('purple')
        ax[3].axis('off')
        ax[4].imshow(green)
        ax[4].set_title('green')
        ax[4].axis('off')
        ax[5].imshow(blue)
        ax[5].set_title('blue')
        ax[5].axis('off')
        ax[6].imshow(grey)
        ax[6].set_title('grey')
        ax[6].axis('off')
        ax[7].imshow(white)
        ax[7].set_title('white')
        ax[7].axis('off')
        ax[8].imshow(black)
        ax[8].set_title('black')
        ax[8].axis('off')
        
        plt.show
 
    return(p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8])