# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 17:12:49 2019

@author: ashok
"""

# -*- coding: utf-8 -*-
"""
## Last amended: 4th July, 2018
## My folder: C:\Users\ashokharnal\OneDrive\Documents\cluster_analysis
## R file: quantization_image.R
## Colour quantization:
## http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py
## https://alstatr.blogspot.in/2014/09/r-k-means-clustering-on-image.html
## https://en.wikipedia.org/wiki/Color_quantization
## http://lmcaraig.com/color-quantization-using-k-means/
## https://rwalk.xyz/color-quantization-in-r/
##
## Objective:
##           To reduce no of colours in a colour palette
##           using k-means

## Steps:
#             1. Read any image (skimage.imread)
#                Shape: 419 X 640 X 3
#             2. Reshape it to: 268160 X 3 (np.reshape)
#             3. Scale color values by dividing by 255
#             4. We have three columns of 270000 rows
#                Group them into 64 clusters (KMeans)
#             5. Find cluster labels of each row (clust_labels)
#             6. Replace each RGB row by its respective
#                cluster-center (model.cluster_centers_[clust_labels])
#             7. Reshape image back and plot it
 
Ref:
    http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py
"""

%reset -f             # rm(list = ls) ; gc()
## 1. Call libraries
import numpy as np
# 1.1. For displaying graphics
import matplotlib.pyplot as plt
# 1.2. For image reading/manipulation
#      Images can be manipulated using opencv, pillow and skimage
#      Install skimage as: 
##     conda install -c anaconda scikit-image
from skimage.io import imread    # Read image      
from skimage.io import imshow    # Display image
from skimage.io import imsave    # Save image
# 1.3 For clustering
from sklearn.cluster import KMeans
# 1.4 OS related
import os
import time  # Measuring process time

# 2. Set working folder and read file
path = "D:\\data\\OneDrive\\Documents\\cluster_analysis"
#os.chdir("C:\\Users\\ashok\\OneDrive\\Documents\\cluster_analysis")
os.chdir(path)

# 2.1 Read the image file
china= imread("china.jpg")
china   # Image is  a multi-dimensional array

# Three colour channels (RGB) of 419 X 640 each
china.shape    # 419 X 640 X 3

### 2.2 Some Experiment on image array
#       Observe some colour values in each frame
china[0,0,0] , china[0,0,1] , china[0,0,2]

# 2.3 What are max and min colour intensites
np.min(china)
np.max(china)

##############################################
# Experiment begins on reshaping image
##############################################
# 2.4 Reshaping image and reshaping back. Is it restored?
#      Extract colour intensity values form 
test = china[120:124, 116:121, 0:3]    
# 2.4.1 Its shape?
test.shape    # (4,5,3)
test          #  Or 4-rows of 5-pixels each
              #   Inner array is RGB coord of each pixel


china[120,116,0]        # 29
china[120,116,1]        # 33
china[121,116,0]        # 10
china[121,116,1]        # 14
china[120,117,0]        # 11
china[120,117,1]        # 15


# 2.4.2 Now reshape it in a 2-d array
test1 = test.reshape(20,3)    # 20 rows X 3 cols

# 2.4.3 Compare the following two: one reshaped
#       and the other not
test1
test

# 2.4.4 And reshape back. Does it compare with original?
test1.reshape(4,5,3)

### Experiment Ends
##############################################

# 3. Reshape china image
newchina = china.reshape(china.shape[0] * china.shape[1],
                         china.shape[2]
                         )
newchina.shape

# 3.1 Normalize all image colors
newchina = newchina/255
newchina.shape

# 3.2 Observe normalized R-G-B colors of top-10 points
newchina[:10, : ]

# 4. Perform clustering of R-G-B
#    Set kmeans parameters. Get 64 colours
# 4.1   Instantiate the class
model = KMeans(n_clusters = 64 )

# 4.2 Perform kmeans clustering (10 minutes)
start = time.time()
clust_labels = model.fit_predict(X = newchina)
end = time.time()
print(end - start)

# 5. Look at cluster labels
clust_labels

# 5.1 How many labels
len(clust_labels)

# 6. And get 64 cluster centers
cent=model.cluster_centers_     # Use model.<tab> to get 'model' attributes
cent
cent.shape

# 6.1 For each cluster label, get RGB values
ff = cent[clust_labels]        #  model.cluster_centers_[clust_labels]
ff.shape


# 7. Get image back by reshaping 
modiImage = ff.reshape(419,640,3)

# 8. Show 64-color image
plt.figure(1)
plt.title('Quantized image (64 colors)')
imshow(modiImage)

# 9. Show original image
plt.figure(2)
plt.title("Original image")
imshow(china)

# 10. Save image and check size. It is reduced.
imsave("modiImage.jpeg", modiImage)


########################################
# http://scikit-learn.org/stable/modules/clustering.html
# http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
def doCluster(X, nclust=64):
    model = KMeans(nclust)
    model.fit(X)
    clust_labels = model.predict(X)
    cent = model.cluster_centers_
    return (clust_labels,cent)

clust_labels,cent = doCluster(newchina,64)
    
############################################

"""
How pixels are arranged in an image:
====================================


For clarity we have taken four rows, five columns and two frames--All different.
The array output is as follows:

china[120:124, 116:121, 0:2]
Out[61]: 
array([[[29, 33],   	|
        [11, 15],   	|
        [ 4,  8],   	| Row 120: five pixels 116,117,118,119,120, Col R & G
        [15, 21],   	|
        [19, 25]],  	|

       [[10, 14],		|
        [ 8, 14],		|
        [12, 18],		| Row 121: five pixels 
        [20, 26],		|
        [17, 23]],	|

       [[27, 28],		|
        [23, 24],		|
        [10, 14],		| Row 122
        [26, 30],		|
        [18, 22]],	|

       [[13, 13],		|
        [15, 15],		|
        [13, 13],		| Row 123
        [15, 15],		|
        [30, 31]]], dtype=uint8)|


china[120,116,0]	=> 29
china[120,116,1]	=> 33
china[121,116,0]  => 10
china[121,116,1]  => 14
china[120,117,0]  => 11
china[120,117,1]  => 15

After reshaping as below, the result is:
test.reshape((5 *5,2))
Out[64]: 
array([[29, 33],
       [11, 15],
       [ 4,  8],
       [15, 21],
       [19, 25],
       [10, 14],
       [ 8, 14],
       [12, 18],
       [20, 26],
       [17, 23],
       [27, 28],
       [23, 24],
       [10, 14],
       [26, 30],
       [18, 22],
       [13, 13],
       [15, 15],
       [13, 13],
       [15, 15],
       [30, 31]], dtype=uint8)

uint8 is unsigned 8-bit integer

"""



