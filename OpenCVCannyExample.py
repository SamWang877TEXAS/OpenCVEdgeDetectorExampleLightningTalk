### OpenCV Example
# Source: https://upload.wikimedia.org/wikipedia/commons/1/10/Front_BMW_Emblem_-_2015_BMW_M3_%2815820478397%29.jpg
# Creative Commons License!

# Packages, make sure to install!
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Reading in the Image
src = cv.imread('Front_BMW_Emblem_-_2015_BMW_M3_(15820478397).jpg')
cv.imshow('Source', src) # Show image
cv.waitKey() # Wait to exit


# Processing
# Greyscale
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
cv.imshow('Grayscale', src_gray)
cv.waitKey()

# Blurred
src_processed = cv.blur(src_gray, (3, 3))
cv.imshow('Blurred', src_processed)
cv.waitKey()

# Canny Edge Detection
# Modify These!
threshold1 = 255
threshold2 = 240
# Find edges/outlines
edges = cv.Canny(src_processed, threshold1, threshold2)
print('Edge Information:')
print(f'Data Type: {type(edges)}')
print(edges)


# Showing the Result
plt.subplot(121), plt.imshow(src) # Note: colors swapped b/c OpenCV stores colors in BGR, but matplotlib does in RGB
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges)
plt.title('Edges Image'), plt.xticks([]), plt.yticks([])
plt.show()



### ADVANCED
# Creating contours (like edges, but connected!)
contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

contours_poly = []
boundRect = []
drawing = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
for i, c in enumerate(contours):
    contours_poly.append(cv.approxPolyDP(c, 3, True))
    boundRect.append(cv.boundingRect(contours_poly[i]))

for i in range(len(contours)):
    cv.drawContours(drawing, contours, i, (255, 255, 255))
    # Below is the bounding rectangle
    # cv.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])),
                 # (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), (255, 255, 255), 2)

cv.imshow('Source', drawing)
cv.waitKey()