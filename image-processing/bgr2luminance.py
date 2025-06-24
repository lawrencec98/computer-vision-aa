# Exercise 3.7 Histogram equalization

# Compute the gray level (luminance) histogram for an image and equalize it so that the tones look
# better (and the image is less sensitive to exposure settings).

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


def bgr2luminance(bgr_image):
    # Luminance can be perceived from RGB by the following equation:
    # 0.2126R + 0.7152G + 0.0722B

    height, width, channels = bgr_image.shape
    lum_img = np.zeros((height,width), dtype=np.uint8)
    
    for row in range(height):
         for col in range(width):
            bval = bgr_image[row][col][0]
            gval = bgr_image[row][col][1]
            rval = bgr_image[row][col][2]

            lum_img[row][col] = (0.0722*bval) + (0.7152*gval) + (0.2126*rval)

    return lum_img


if __name__ == "__main__":
    # Read in an image
    rgb_img = cv.imread('rgb_tucan.jpg')

    # Convert the image to greyscale (luminance)
    luminance_image = bgr2luminance(rgb_img)

    cv.imshow('before',rgb_img)
    cv.imshow('after', luminance_image)
    cv.waitKey(0)