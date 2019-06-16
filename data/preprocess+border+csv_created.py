# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:12:05 2019

@author: s162152
"""
from borderirregularity import compactness,abruptness,convexity
from preprocessing import *
import cv2
import os
from pathtofiles() import *
import pandas as pd

def main():
    featurelist=[]
    ##change this to the path where the images are saved
    dir_input=pathtofiles()+'/image_data'
    
    ##import the gathered data of the given masks
    df = pd.read_csv(pathtofiles()+'csvfiles/output.csv')
    mask_ID=df['image']
    mask_comp=df['compactness']
    mask_abrupt=df['abruptness']
    mask_conv=df['convex']

    ##import the file with the images for which yields that the given masks need to be used
    old=pd.read_csv(pathtofiles()+'csvfiles/use_given_masks.csv')
    use_old_masks=list(old['image'])

# loop through *.jpg files in the current directory, preprocess the images and gather the data
    for f in os.listdir(dir_input):
        if os.path.isfile(f) and f[-8:] == "0100.jpg":
            y=str(f)
            img = cv2.imread(f,0)
            ##add the values of the given masks
            if y in use_old_masks:
                    y.split()
                    ID=y[0:12]+str('_segmentation.png')
                    for i in range(len(mask_ID)):
                         if mask_ID[i]==ID:
                             compact=mask_comp[i]
                             abrupt=mask_abrupt[i]
                             convex=mask_conv[i]
            ## gather data with the created masks
            else:
                image=preprocessing(img)
                compact=compactness(image)
                abrupt=abruptness(image)
                convex=convexity(image)
            featurelist.append(y+','+str(compact)+','+str(convex)+','+str(abrupt))
    ##create csv file in a folder called csvfiles
    with open(pathtofiles()+'csvfiles/output_created_masks.csv', 'w',newline='') as csvfile:
        fieldnames = ['image','compactness', 'convex', 'abruptness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in featurelist:
            h=i.split(",")
            writer.writerow({'image': h[0], "compactness":h[1], 'convex': h[2], 'abruptness':h[3]})
                        
    
    
if __name__ == "__main__": main()
