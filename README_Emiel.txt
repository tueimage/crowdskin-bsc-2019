# Melanoma Classification based on Automated Color Detection

In this project, I built an automated detection algorithm to detect different colors in a skin lesion.
I did this using two methods:
	(1) Color variegation
	(2) k-Means clustering
The outcome of the automated detection is used for binary classification of skin lesions. 
Two simple classifiers are used for the classification task:
	(1) Logistic regression
	(2) k-Nearest Neighbor (kNN)
The performance outcome will be compared to the performance of classification on crowd-sourced annotations of the ABC-sings
done by first-year bachelor students. 

To run the script, first, you have to change the paths in config.py.
### config.py:
	LESION_PATH	: location of ISIC database
	MASK_PATH	: location of ISIC segmentation masks
	GROUPS_PATH	: location of excel files with crowd annotations
	MAIN_PATH	: location of scripts and final data

### run_Variegation.py:
Runs the color variegation algorithm for all images in the ISIC database by calling the color.py.
Saves the shares of the nine final colors incl. image name and ground truth classification in a csv file.
Indicate if you want plots for verification of color determination. (Not recommended if you run all images)

### color.py
Contains the color variegation algorithm.
Input is image name (e.g. ISIC_0000000) and plots=0/1.
Returns shares of nine final colors.

### Cluster_kMeans.py
Calculates the Sum of Squared Distances and the correlation coefficients between them.
Returns a csv file with first four correlation coefficients incl. image name and ground truth classification

### raincloud.py
Displays the shares obtained with run_Variegation.py in raincloudplots.
Only possible per color.

### Logistic_Regression.py
Reads the shares and normalize the features. 
Performs logistic regression for dataset. Indicate in MAIN() which csv file you want.

### CM_ROC.py
Contains formula for AUC.
Able to plot Confusion Matric and ROC curve.

### kNN_.py
Self written k-Nearest-Neighbor algorithm, with Eucledian distance and Chi-Square as
distance meausure. Adjustments needed if you want to use it for other datasets.

### Features_Combined.py
Combines my features with the features of Audrey and Sanne.
Performs logistic regression with these features.

### Crowd_LogReg.py
Opens the crowd annotations group by group.
Averages the annotations per group over the number of annotators.
To be able to merge the groups, the averaged annotations are scaled.
Logistic regression is performed on the merged dataset. (in line 86 able to select features: ABC)

### PCA_.py
Principal Component analysis performed on annotated ABC-signs.
