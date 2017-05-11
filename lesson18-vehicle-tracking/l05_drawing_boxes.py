import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    '''Draw a box around bboxes points in img'''
    draw_img = np.copy(img)
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    return draw_img

if __name__ == '__main__':
    image = mpimg.imread('bbox-example-image.jpg')

    bboxes = [((275, 572), (380, 510)), ((488, 563), (549, 518)), ((554, 543), (582, 522)), 
              ((601, 555), (646, 522)), ((657, 545), (685, 517)), ((849, 678), (1135, 512))]

    result = draw_boxes(image, bboxes)
    plt.imshow(result)
    plt.show()
