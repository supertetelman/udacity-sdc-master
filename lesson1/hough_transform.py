'''Small Python script created during Udacity SDC-NN lesson 1 lectures/labs'''


import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


img_name = 'exit-ramp.jpg'
res_dir = 'res/'
data_dir = 'data/'
image = mpimg.imread(data_dir + img_name)


# Create several images that will be used to demonstrate all the steps
# in this pipeline along with different results had features like
# blurring not been used.
line_image = np.copy(image)*0 #creating a blank to draw lines on
blur_line_image = np.copy(image)*0 #creating a blank to draw lines on
masked_line_image = np.copy(image)*0 #creating a blank to draw lines on
masked_blur_line_image = np.copy(image)*0 #creating a blank to draw lines on
region_select = np.copy(image)*0


# Configurable Parameters
kernel_size = 3 # Gaussian Blur kernel size
ratio = 3 # Canny low:high threshold ratio
low_threshold = 70 # Canny Low threshold
high_threshold = ratio * low_threshold  # Canny high threshold

# Define the Hough transform parameters
rho = 1
theta = np.pi/155
threshold = 185
min_line_length = 10
max_line_gap = 10


# Note that from here foward we maintain seperate image copies at each step
# For a blurred and unblurred image. This is to explore the effects of blurring
# on the final result.
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #grayscale conversion 
edges = cv2.Canny(gray, low_threshold, high_threshold)
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
blur_edges = cv2.Canny(blur_gray, low_threshold, high_threshold)    


# Use the HoughLinesP Function and our parameters to connect all the points
# we decided qualify as lines and return a list of lines in format [x1,y1,x2,y2]
def do_hough(edges):
    '''Take a list of edges, and use global parameters to generate a list of lines'''
    return  cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
            min_line_length, max_line_gap)


lines = do_hough(edges)
blur_lines = do_hough(blur_edges)


# Render lines for our four main images
def draw_lines(image, lines):
    '''For each detected line in lines, render a straight line on image'''
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(image,(x1,y1),(x2,y2),(255,0,0),10)	


draw_lines(line_image, lines)
draw_lines(blur_line_image, blur_lines)


# Stack all the XxYx1 grayscale edge images to create XxYx3 RGB color images
def stack_edges(edges):
    return np.dstack((edges, edges, edges))


color_edges = stack_edges(edges)
blur_color_edges = stack_edges(blur_edges)


# Overlay the drawn lines colored line_image over the 
def overlay_edges(edge_img, line_img):
    return cv2.addWeighted(edge_img, 0.8, line_img, 1, 0)


combo = overlay_edges(color_edges, line_image) 
blur_combo = overlay_edges(blur_color_edges, blur_line_image) 
lines_edges = overlay_edges(image, blur_line_image)


# Save all our images to files
mpimg.imsave(res_dir + img_name +'.gray.jpg', gray)
mpimg.imsave(res_dir + img_name +'.gray-blur.jpg', blur_gray)
mpimg.imsave(res_dir + img_name +'.edges.jpg', edges)
mpimg.imsave(res_dir + img_name +'.edges-blur.jpg', blur_edges)
mpimg.imsave(res_dir + img_name +'.edges-color.jpg', color_edges)
mpimg.imsave(res_dir + img_name +'.edges-color-blur.jpg', blur_color_edges)
mpimg.imsave(res_dir + img_name +'.hough.jpg', combo)
mpimg.imsave(res_dir + img_name +'.hough-blur.jpg', blur_combo)
mpimg.imsave(res_dir + img_name +'lanes.jpg', lines_edges)
