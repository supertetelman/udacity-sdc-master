import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


def abs_sobel_thresh(img, orient='x', thresh=(0,255), sobel_kernel = 3):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take derif in respect to orient
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
        
    # Take absolute value
    sobel = np.absolute(sobel)
    
    # Scale to 0-255 and cast to int
    sobel = 255 * sobel / np.max(sobel)
    sobel = np.uint8(sobel)
    
    # Create a mask with 1s where the threshold is met
    mask = np.zeros(gray.shape)
    mask[(sobel > thresh[0]) & (sobel < thresh[1])] = 1
    
    return mask


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Generate sobel
    x_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    y_sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    
    # Calculate scaled magnitude
    magnitude = np.sqrt(x_sobel ** 2 + y_sobel ** 2)
    scaled_magnitude = 255 * magnitude / np.max(magnitude)
    
    # Mask based on threshold
    mask = np.zeros(gray.shape)
    mask[(scaled_magnitude > mag_thresh[0]) & (scaled_magnitude < mag_thresh[1])] = 1
    
    return mask


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Take sobel gradients
    x_sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    y_sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    abs_x = np.absolute(x_sobel)
    abs_y = np.absolute(y_sobel)
    
    # Calculate direction
    direction = np.arctan2(abs_y, abs_x)
    
    # Create a threshold based mask
    mask = np.zeros(gray.shape)
    mask[(direction > thresh[0]) & (direction < thresh[1])] = 1

    return mask


# Read in an image
image = mpimg.imread('signs_vehicles_xygrad.png')

# Choose a Sobel kernel size
ksize = 3 # Choose a larger odd number to smooth gradient measurements
dir_thresh = (0, np.pi/2)
mag_thresh = (0, 255)
x_thresh = (0, 255) 
y_thresh = (0, 255)

# Apply each of the thresholding functions
gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=x_thresh)
grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=y_thresh)
mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=mag_thresh)
dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=dir_thresh)


# Create some combinations
x_y = np.zeros(gradx.shape)
x_y[(grady == 1) & (gradx == 1)] = 1

mag_dir = np.zeros(gradx.shape)
mag_dir[(mag_binary == 1) & (dir_binary == 1)] = 1

x_y_mag = np.zeros(gradx.shape)
x_y_mag[(gradx == 1) & (grady == 1) & (mag_binary == 1)] = 1

x_y_dir = np.zeros(gradx.shape)
x_y_dir[(gradx == 1) & (grady == 1) & (dir_binary == 1)] = 1

x_mag = np.zeros(gradx.shape)
x_mag[(gradx == 1) & (mag_binary == 1)] = 1

x_dir = np.zeros(gradx.shape)
x_dir[(gradx == 1) & (dir_binary == 1)] = 1

y_mag = np.zeros(gradx.shape)
y_mag[(grady == 1) & (mag_binary == 1)] = 1

y_dir = np.zeros(gradx.shape)
y_dir[(grady == 1) & (dir_binary == 1)] = 1

all_thresh = np.zeros(gradx.shape)
all_thresh[(gradx == 1) & (grady == 1 ) & (mag_binary == 1) & (dir_binary == 1)] = 1

# Display everything  # TODO: Make these all subplots
plt.imshow(gradx, cmap='gray')
plt.title("x gradient")
plt.show()

plt.imshow(grady)
plt.title("y gradient")
plt.show()

plt.imshow(mag_binary)
plt.title("magnitude")
plt.show()

plt.imshow(dir_binary)
plt.title("directional")
plt.show()

plt.imshow(x_y)
plt.title("x and y gradient")
plt.show()

plt.imshow(mag_dir)
plt.title("magnitude and direction")
plt.show()

plt.imshow(x_mag)
plt.title("x gradient and magnitude")
plt.show()

plt.imshow(x_dir)
plt.title("x gradient and direction")
plt.show()

plt.imshow(y_mag)
plt.title("y gradient and magnitude")
plt.show()

plt.imshow(y_dir)
plt.title("y gradient and direction")
plt.show()

plt.imshow(x_y_mag)
plt.title("x and y gradient and magnitude")
plt.show()

plt.imshow(x_y_dir)
plt.title("x and y gradient and direction")
plt.show()

plt.imshow(all_thresh)
plt.title("all gradients")
plt.show()
