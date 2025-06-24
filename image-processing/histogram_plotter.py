# Exercise 3.7 Histogram equalization

# Compute the gray level (luminance) histogram for an image and equalize it so that the tones look
# better (and the image is less sensitive to exposure settings).

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


###############################
# Step 1: Convert the color image to luminance.

# Read image
rgb_img = cv.imread('rgb_tucan.jpg')
temp = cv.cvtColor(rgb_img, cv.COLOR_BGR2GRAY) # Convert from BGR to Grayscale

mono_img = temp.copy()

# Display images
# cv.imshow('before', rgb_img)
# cv.imshow('after', mono_img)
# if cv.waitKey(0) == 'q':
# 	cv.destroyAllWindows()


###############################
# Step 2: Compute the histogram, the cumulative distribution, and the compensation transfer function

intensity_values = np.array(0)

for row in mono_img:
	for pixel in row:
		intensity_values = np.append(intensity_values,pixel)


# Plot histogram and cumulative distribution
fig, ax1 = plt.subplots()

counts, bins, patches = ax1.hist(intensity_values, bins=260, density=False, alpha=0.75, label='Histogram')
ax1.set_xlabel('Pixel Intensity')
ax1.set_ylabel('Count')
ax1.legend(loc='upper left')

# Cumulative on second y-axis
ax2 = ax1.twinx()
ax2.hist(intensity_values, bins=260, density=True, histtype='step', cumulative=True, color='r', label='Cumulative')
ax2.set_ylabel('Cumulative Density')
ax2.legend(loc='upper right')

plt.title("Histogram and Cumulative Distribution")
plt.grid(True)
plt.show()