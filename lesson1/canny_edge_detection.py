'''Small Python script created during Udacity SDC-NN lesson 1 lectures/labs'''

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

img_name = 'exit-ramp.jpg'
res_dir = 'res/'
data_dir = 'data/'
image = mpimg.imread(data_dir + img_name)

def do_canny_edge(image, ratio, low_threshold, kernel_size, final_iter=False):
    '''Take an image, low-to-high Canny threshold value, blur kernel size
    and a flag indicating to print all or final image.
    Saves images to specified file and returns the final image
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion    

    # Define a kernel size for Gaussian smoothing / blurring
    # Note: this step is optional as cv2.Canny() applies a 5x5 Gaussian internally
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
    high_threshold = ratio * low_threshold    

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    blur_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)    
    
    # the setting of low_threshold will have the most variance
    # use that in the filename first for better analysis
    param_str = ".%d-%d-%d"%(low_threshold, ratio, kernel_size)
    
    # Prevent all images from being saved during the experimental phase using a flag
    if final_iter:
        mpimg.imsave(res_dir + img_name + param_str + '.gray.jpg', gray)
        mpimg.imsave(res_dir + img_name + param_str + '.gray-blur.jpg', blur_gray)
        mpimg.imsave(res_dir + img_name + param_str +'.edges.jpg', edges)
    mpimg.imsave(res_dir + img_name + param_str +'.edges-blur.jpg', blur_edges)
    return blur_edges


def test_canny_params():
    '''Quick function to visually check the affects
     of GaussianBlur and cv2.Canny parameters.
     '''
    #for ratio in [2, 3, 4]:
    for ratio in [3, 4]:
        #for kernel_size in [3, 5, 9]:
        for kernel_size in [3, 5]:
            # for threshold in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
            for threshold in [70, 90, 120]:
                do_canny_edge(image, ratio, threshold, kernel_size)


#test_canny_params()
do_canny_edge(image,3, 70, 3, True) # I found 3, 70, 3 to give the best results