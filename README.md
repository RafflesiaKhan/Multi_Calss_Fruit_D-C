# Multi_Calss_Fruit_D-R (FDR)

# Description: 
Combining object detection and recognition approaches, we have developed a competent multi-class Fruit Detection and Recognition (FDR) model that is very proficient regardless of different limitations such as high and poor image quality, complex background or lightening condition, different fruits of same shape and color, multiple overlapped fruits, existence of non-fruit object in the image and the variety in size, shape, angel and feature of fruit. This proposed FDR model is also capable of detecting every single fruit separately from a set of overlapping fruits. Another major contribution of our FDR model is that it is not a dataset oriented model which works better on only a particular dataset as it has been proved to provide better performance while applying on both real world images (e.g., our own dataset) and several states of art datasets.

  Fruit_Segmentation folder contains the code for segmentation ( detecting and segmenting fruits from input image ).

  Fruit_Recognition folder contains the code for classification ( recognizing the class of each segmented fruit).

# How to run
  -  At folder Fruit_Segmentation search for main file named 'TRy2_Copy.py' it tooks input from Fruit_Segmentation/input directory and performs segmentation.
  -  Running 'TRy2_Copy.py' will show the results of segmentation.
  -  The most promising results would be stored at Fruit_Segmentation/Test1 
  -  At folder Fruit_Recognition search for main file named 'DataTest5_55.ipynb' which performs the classification.
  -  'DataTest5_55.ipynb' contatins the description fro each cell.
  -  One can choose their own Training and Test set for classification.
  -  The objective here is to get the segmentation results as Test input for classification. 
  
 > This Project is still under development, it will get updated as soon as new versions develop.
