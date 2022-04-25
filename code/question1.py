import cv2
import numpy as np

# converting image to grayscale
raw_image = cv2.imread('Q1image.png')
gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)

# making a kernel that looks like a disk to retain circular shapes and recreate circular shapes in morphology
disk_kernel = np.ones((5, 5), np.uint8)
disk_kernel[0, 0], disk_kernel[0, 4], disk_kernel[4, 0], disk_kernel[4, 4] = 0, 0, 0, 0
disk_kernel = disk_kernel.astype(np.uint8)

# eroding and then dilating the image to get 'disk' shapes that are separated from each other
eroded = cv2.erode(gray, disk_kernel, iterations=10)
dilated = cv2.dilate(eroded, disk_kernel, iterations=8)

# finding number of connected components wth the connectedComponents() function
# black background counts as a component so subtracting 1 from answer
connected_components = cv2.connectedComponents(dilated)
coins = connected_components[0] - 1

# displaying answers for part 1 and 2
print(f'Part 2: Number of coins is {coins}')
cv2.imshow('Part 1', dilated)
cv2.waitKey(0)
