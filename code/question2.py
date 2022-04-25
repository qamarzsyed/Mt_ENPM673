import cv2
import numpy as np

# importing both images and detecting all features
# originally planned to use SIFT, but according to OpenCV that is currently patented and ORB didn't work well so BRISK
image_a = cv2.imread('Q2imageA.png')
image_b = cv2.imread('Q2imageB.png')
detect = cv2.BRISK_create()

kp_a, coord_a = detect.detectAndCompute(image_a, None)
kp_b, coord_b = detect.detectAndCompute(image_b, None)


# Brute force matcher to match features to each other from different perspectives
bfm = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
matches = bfm.match(coord_a, coord_b)
test = cv2.drawMatches(image_a, kp_a, image_b, kp_b, matches, None)

# selects the best 25 matches and puts them into an array of a and b points
n = 25
if n > len(matches):
    n = len(matches)
top_matches = sorted(matches, key=lambda x: x.distance)[:n]

a_points = np.array([kp_a[match.queryIdx].pt for match in top_matches])
b_points = np.array([kp_b[match.trainIdx].pt for match in top_matches])

# finds homography matrix, warps images, and then stitches one on top of the other
H, stat = cv2.findHomography(b_points, a_points)
warped = cv2.warpPerspective(image_b, H, (image_a.shape[1]*5//3, image_a.shape[0]*4//3))
warped[0:image_a.shape[0], 0:image_a.shape[1]] = image_a


cv2.imshow('Stitched Image', warped)
cv2.waitKey(0)
