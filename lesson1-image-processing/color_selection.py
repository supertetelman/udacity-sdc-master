''' Simple python script that takes an image and blacks out non-white pixels
Used as the first step in lane detection. 
Created during Udacity SDC-NN lesson 1 lectures/labs
'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


# Read in the image
res_dir = 'res/'
data_dir = 'data/'
img_name = 'test.jpg'
image = mpimg.imread(data_dir + img_name)


# Grab the x and y size and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
color_select = np.copy(image)


# Define color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200


rgb_threshold = [red_threshold, green_threshold, blue_threshold]


# Do a boolean or with the "|" character to identify
# pixels below the thresholds
thresholds = (image[:,:,0] < rgb_threshold[0]) \
        | (image[:,:,1] < rgb_threshold[1]) \
        | (image[:,:,2] < rgb_threshold[2])
color_select[thresholds] = [0,0,0]
plt.imshow(color_select)
# Display the image                 
plt.imshow(color_select)


mpimg.imsave(res_dir + img_name + '.color-post.jpg', color_select)
