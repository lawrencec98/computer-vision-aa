import cv2 as cv
import numpy as np
import bgr2luminance as bl


# Simple 3x3 gaussian filter
# 1   2   1
# 2   4   2
# 1   2   1


# Manually set the gaussian coefficients, then normalize
gaussian_filter = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16


# Convolution
def correlation(input_image, filter):

    image = input_image.copy()

    height, width = image.shape

    filtered_image = np.zeros((height, width), np.uint8)

    for row in range(1, height-1):
            for col in range(1, width-1):
                total = 0
                for i in range(3):
                    for j in range(3):
                        total += filter[i][j] * image[row-1+i][col-1+j]
                
                filtered_image[row][col] = int(total)

    return filtered_image


if __name__ == "__main__":

    original_image = cv.imread('rgb_tucan.jpg')
    mono_image = bl.bgr2luminance(original_image)

    filtered_image = correlation(mono_image, gaussian_filter)

    cv.imshow('original', mono_image)
    cv.imshow('filtered', filtered_image)
    cv.waitKey(0)