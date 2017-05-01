import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def corners_unwarp(img, nx, ny, mtx, dist):
    # Undistort image
    undistort = cv2.undistort(img, mtx, dist)

    # Convert to grayscale
    gray = cv2.cvtColor(undistort, cv2.COLOR_BGR2GRAY)

    # Find corners
    corners_ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    
    # Verify that all corners were found
    assert corners_ret, "All corners could not be found."
    
    # Draw corners on the image
    img = cv2.drawChessboardCorners(undistort, (nx, ny), corners, corners_ret)

    # Define 4 well identified outer corners 
    src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    
    # Define 4 destination points offset slightly from the image corners
    img_size = (gray.shape[1], gray.shape[0])
    offset = 100
    dst = np.float32([[offset, offset],
                      [img_size[0]-offset, offset], 
                      [img_size[0]-offset, img_size[1]-offset], 
                      [offset, img_size[1]-offset]])
    
    # Get perspective tranformation matrix M
    M = cv2.getPerspectiveTransform(src, dst)
    
    # Transform the perspective
    warped = cv2.warpPerspective(undistort, M, img_size)    

    return warped, M


# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

# Read in an image
img = cv2.imread('test_image2.png')
nx = 8 # the number of inside corners in x
ny = 6 # the number of inside corners in y
top_down, perspective_M = corners_unwarp(img, nx, ny, mtx, dist)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(top_down)
ax2.set_title('Undistorted and Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
