# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:12:05 2019

@author: s162152
"""
from borderirregularity import compactness,abruptness,convexity
from preprocessing import *
import cv2
import os
import pandas as pd
from pathtofiles import *

def main():
    featurelist=[]
    ##Change this to the folder where the masks from the ISIC challenge are located
    dir_input=pathtofiles()+'ISIC-2017_Training_Part1_GroundTruth'
    ## loop through *.jpg files in the current direcotry and calculate the different border methods
    for f in os.listdir(dir_input):
        if os.path.isfile(f) and f[-17:] == "_segmentation.png":
            y=str(f)
            img = cv2.imread(f,0)
            compact=compactness(img)
            abrupt=abruptness(img)
            convex=convexity(img)

            featurelist.append(y+','+str(compact)+','+str(convex)+','+str(abrupt))
            print(featurelist)
    ## Create a csv file of the outcome of the different border methods
    with open(pathtofiles()+'csvfiles/output.csv', 'w',newline='') as csvfile:
        fieldnames = ['image','compactness', 'convex', 'abruptness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in featurelist:
            h=i.split(",")
            writer.writerow({'image': h[0], "compactness":h[1], 'convex': h[2], 'abruptness':h[3]})
                        
    
    
if __name__ == "__main__": main()
