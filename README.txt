In this project classification based on automated methods and crowd annotations is done focussing on asymmetry. 
The automated asymmetry measurement is done in two ways, both using the already existing masks from the ISIC-2017 challenge. 
The annotations are obtained from first year Biomedical Enigneering. 

To be able to run the codes, you have put the codes in the folder with the masks of the 
skin lesions e.g. in a folder named image_data. The csvfile for the asymmetry data will also be 
saved in this folder. The csv files and the excel files that will be used in the codes must also be put in this folder. 
To be able to run the main code (main.py), you need to alter line 21 in your own path. 

For the asymmetry measurments, I created two scripts; asymmetry1.py and asymmetry2.py 
### Asymmetry.py:
It determines the asymmetry of mask via method 1. 

### Asymmetry2.py: 
It determines the asymmetry of a mask via method 2. 

### main.py:
This scripts determines the asymmetry of all masks present in the folder via both asymmetry 
methods. It saves the results of both methods in a csv file called datafinal.csv. 
In this file are also the ground truth labels saved. These are extracted from the file 
ISIC-2017_Training_Part3_GroundTruth.csv

For the analysis of the asymmetry results I created the following script:
### analyze_asym.py:
This function must be applied to the file datafinal.csv. 
In this script several other function are called from the file functions.py:
	## shapiro: 
	This function perfomes a Shapiro-Wilk test on the given data set and will return 
	P-values of both asymmetry methods.
	## ztest: 
	This function performes a Z-test on the given data set. It will give the P-value and 
	it will tell you to reject the null hypothesis when this value is below 0.05.
	## meanANDstd:
	This function will determine the mean and standard deviation of both asymmetry methods 
	## RainCloud:
	This function will visualize the two asymmetry methods by using a rain cloud plot. 
This script also visualizes the annotations by making a rain cloud plot. To do this, all groups are
merged. 

The classification of the skin lesions is done in two ways: 

### Logistic_final.py:
This script performs logistic regression on different data sets (annotations, method 1, method 2, method 1 and 2,
all ABC features). It performs a 10-fold-cross validation and will return the mean AUC value per method. This script
also determines the mean sensitivy and specificity per method. The function will also visualize the AUC results 
by making boxplots of the different methods. 

### knn_final.py:
This script perfoms knn on different data sets with a k of five. It performs a 10-fold-cross validation and will return the mean AUC value per method. This script
also determines the mean sensitivy and specificity per method. The function will also visualize the AUC results 
by making boxplots of the different methods.  