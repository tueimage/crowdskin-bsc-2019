In this project segmentation, feature extraction, classification and visualization/analayzing has been done.
To do the segmentation, I used two methods. One method with the already existing masks from the ISIC-2017 challenge 
in this project referred to as the given masks and one method based on the biggest contour, in this project referred to as created masks.
However the method based on the biggest contour couldn't be used for all images.

To be able to run the code, you have to adapt pathtofiles.py to your own path where the data is saved
create a folder for your data. In this folder, add folders named
csvfiles: here the csvfiles will be saved
image_data:  This folder contains the images from the ISIC-2017 challenge
groups: this folder contains the data of the different groups
ISIC-2017_Training_Part1_GroundTruth: This folder contains the binary masks of the ISIC-2017 challenge



To decide whether that segmentation method could be used, I created the mask_or_not.py file.

###mask_or_not.py file:
It decides whether the lesion area extracted by the segmentation method touches the border of the image.
If that is the case, the image is added to a csv file, named: use_given_masks.csv


###For the feature(border irregularity) extraction, 
the code: preprocess+border+csv_created.py or preprocess+border+csv_given.py 
is used depending on which segmentation method you want to use.

In this code several other functions are called:
Only in the code for the created segmentation, the preprocessing function from preprocessing.py is called 
In this function the image is blurred and tresholded and then a mask is created based on the biggest contour.
Then the different border methods are used to determine the automated values of border irregularity.
The functions of the different border methods: compactness, convexity and abruptness are defined in
borderirregularity.py
Lastly a csv file is created of the values for each image.

###To visualize the data a rain plot cloud is made with the function: raincloudplot.py
In this function, one has to choose whether one wants the visualization of the data of the
given or created masks. 
For the given masks; one needs to use line 50 and alter line 77 to str('_segmentation.png')
instead of str('.jpg').
For the created masks; Use line 51 and for line 77:use str('.jpg')

###To classify and show the results the code: boxplot.py is used
The classification is done with logistic regression and svm as classifiers
In this code, one has to make two choices:
- mask: created or given
  Do you want to classify the data based on the created or given masks
-option: visual, 2000 or 2000all
  Do you want the classification and the boxplot to be of:
	visual:individual border methods, combination of border methods and the visual scores(around 700 images)
	2000: individual border methods and a combination of border methods(around 2000 images)
	2000all: combination of border methods, combination of asymmetry methods, combination of color methods and a combination of all methods(2000 images)

In this code two functions are called:
-classify_visual
	which does the classification based on visual scores of the border
-classify_automatic
	which does the classification based on your choice of combination between automated methods
