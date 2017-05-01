import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # Convert to HLS
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Pull out the Saturation Channel
    s_channel = hls[:,:,2]

    # Mask off a threshold on th Saturation
    output = np.zeros_like(s_channel)
    output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    
    return output
    

# Read in an image, you can also try test1.jpg or test4.jpg
image = mpimg.imread('bridge_shadow.jpg') 

hls_binary = hls_select(image, thresh=(90, 255))

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(hls_binary, cmap='gray')
ax2.set_title('Thresholded S', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
