import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def bin_spatial(img, color_space='RGB', size=(32, 32)):

    color_map = {
            'HSV': cv2.COLOR_RGB2HSV,
            'LUV':cv2.COLOR_RGB2LUV,
            'HLS':cv2.COLOR_RGB2HLS,
            'YUV':cv2.COLOR_RGB2YUV,
            'YCrCb':cv2.COLOR_RGB2YCrCb,
            'Gray': cv2.COLOR_RGB2GRAY
    }

    # Convert image to new color space (if specified)
    if color_space in color_map:
        feature_image = cv2.cvtColor(img, color_map[color_space])
    else:
        feature_image = np.copy(img)
    features = cv2.resize(feature_image, size).ravel()

    return features
    
if __name__ == '__main__':

    # Read in an image
    image = mpimg.imread('cutout1.jpg')

    feature_vec = bin_spatial(image, color_space='HLS', size=(32, 32))

    # Plot features
    plt.plot(feature_vec)
    plt.title('Spatially Binned Features')
    plt.show()
    