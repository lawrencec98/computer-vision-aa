# Implement Gaussian filter
# Take a blurry or nosy image and try to improve the appearance and legibility

import cv2 as cv
import numpy as np

import bgr2luminance as bl


def makenoisy(image):

    # Make image noisy
    noisy_im = image.copy()

    ## Create noise matrix
    shape = noisy_im.shape

    height, width = shape

    ## Add noise matrix to original image
    for row in range(height):
        for col in range(width):
            noise = np.random.randint(0, 25)
            new_val = max(0, min(255, noisy_im[row][col] + noise))
            noisy_im[row][col] = max(0, min(255, new_val))

    return noisy_im



if __name__ == "__main__":
    im = cv.imread('rgb_tucan.jpg')
    im = bl.bgr2luminance(im)

    noisy_image = makenoisy(im)

    # Display
    cv.imshow('original image', im)
    cv.imshow('noisy image', noisy_image)

    cv.waitKey(0)