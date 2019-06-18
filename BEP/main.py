# -*- coding: utf-8 -*-
"""
Created on Mon May 20 14:09:48 2019

@author: s163729
"""

def main():
    #measure the asymmetry from all images and save the results in a csv file
    from asymmetry2 import asymmetry2
    from asymmetry import asymmetry 
    import glob 
    import csv
    
    asym1=[]
    asym2 =[]
   
    melanoom_truth = []

    imageID= []
    for file in glob.glob('/Users/s163729/Documents/BEP/image_data/*png'):
        asym1.append(float(asymmetry(file)))
        asym2.append(float(asymmetry2(file)))

    
    #open ground truth file 
    with open('ISIC-2017_Training_Part3_GroundTruth.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader,None) #skip headers 
        for lines in reader:
            #print(lines)
            imageID.append(str(lines[0]))
            melanoom_truth.append(str(lines[1]))    
    #save results in csv file 
    with open('datafinal.csv', 'w', newline='') as f:
        fieldnames = [ 'imageID', 'asymmetry1', 'asymmetry2', 'melanoma_truth']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for val in range(len(asym1)):
            writer.writerow({'imageID': imageID[val], 'asymmetry1': asym1[val], 'asymmetry2': asym2[val], 'melanoma_truth': melanoom_truth[val]})

if __name__ == "__main__":
    main()