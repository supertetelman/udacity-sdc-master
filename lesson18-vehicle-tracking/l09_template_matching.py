import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from l05_drawing_boxes import draw_boxes

    
def find_matches(img, template_list):
    '''Given an image and a list of template image draw a box around matches'''
    bbox_list = []
    # Define matching method
    # Other options include: cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCORR',
    #         'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED'
    method = cv2.TM_CCOEFF_NORMED
    for temp in template_list:
        tmp = mpimg.imread(temp)
        result = cv2.matchTemplate(img, tmp, method) # to search the image
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result) # extract the location of the best match

        # Determine a bounding box for the match
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        bbox_list.append((top_left, bottom_right))
    return bbox_list

if __name__ == '__main__':
    image = mpimg.imread('bbox-example-image.jpg')
    image2 = mpimg.imread('temp-matching-example-2.jpg')
    templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg',
                'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']


    bboxes = find_matches(image, templist)
    result = draw_boxes(image, bboxes)
    plt.imshow(result)
    plt.title("Identify cars that much templates.")
    plt.show()

    print()
    bboxes = find_matches(image2, templist)
    result = draw_boxes(image2, bboxes)
    plt.imshow(result)
    plt.title("Same template with small time progression - should fail.")
    plt.show()
