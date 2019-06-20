# -*- coding: utf-8 -*-
"""
This script imports color.py and calculates for each image in het ISIC data the
share of nine pre-selected colors in the skin lesion.
Possible to make plots of all nine colors mask by change 'plots' in line 17 to 1.

@author: s151385
"""

import pandas as pd
import os
import numpy as np
import color
import csv
from config import LESION_PATH as LESION_PATH
from config import MASK_PATH as MASK_PATH
from config import MAIN_PATH as MAIN_PATH

### 1 = yes, 0 = no
plots = 0

melanoom_truth = []
image_names = []   
#open ground truth file 
os.chdir(MAIN_PATH)
with open('ISIC-2017_Training_Part3_GroundTruth.csv', 'r') as file:
    reader = csv.reader(file, delimiter=',')
    next(reader,None) #skip headers 
    for lines in reader:
        image_names.append(str(lines[0]))
        melanoom_truth.append(str(lines[1]))           

lijst = np.zeros([len(image_names),9])
for i in range(0,2000):
    image_name = image_names[i]
    p1,p2,p3,p4,p5,p6,p7,p8,p9 = color.colorcoding(image_name,plots)
    print(i)
    lijst[i,0] = p1
    lijst[i,1] = p2
    lijst[i,2] = p3
    lijst[i,3] = p4
    lijst[i,4] = p5
    lijst[i,5] = p6
    lijst[i,6] = p7
    lijst[i,7] = p8
    lijst[i,8] = p9
    
#save results in csv file 
os.chdir(MAIN_PATH)
with open('data.csv', 'w', newline='') as f:
    fieldnames = [ 'imageID', 'p1','p2','p3','p4','p5','p6','p7','p8','p9', 'melanoom_truth']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for val in range(len(lijst)):
        writer.writerow({'imageID': image_names[val], 'p1': lijst[val,0],'p2': lijst[val,1],
                         'p3': lijst[val,2],'p4': lijst[val,3],'p5': lijst[val,4],'p6': lijst[val,5],'p7': lijst[val,6],'p8': lijst[val,7],'p9': lijst[val,8],'melanoom_truth': melanoom_truth[val]})
