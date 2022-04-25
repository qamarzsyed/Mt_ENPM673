import numpy as np
import cv2

# number of clusters and initial image
K = 4
image = cv2.imread('Q4image.png')

# function to evaluate euclidean distances
def distance(color1, color2):
    b_c, g_c, r_c = color1
    b, g, r = color2
    return pow(pow(b_c - b, 2) + pow(g_c - g, 2) + pow(r_c - r, 2), 0.5)

# cluster class, holds mean and previous mean
class Cluster:
    def __init__(self):
        self.mean = [np.random.randint(256), np.random.randint(256), np.random.randint(256)]
        self.count = 1
        self.b_total, self.g_total, self.r_total = self.mean
        self.prev = [-100, -100, -100]

    def add(self, color):
        self.count += 1
        b, g, r = color
        self.b_total += b
        self.g_total += g
        self.r_total += r
        self.mean = [self.b_total//self.count, self.g_total//self.count, self.r_total//self.count]
        return self.mean


# defining initial variables and cluster list and finished cluster list to start loop
h, w, d = image.shape
clusters = [Cluster() for counter in range(K)]
finished = []

# loop while there is a cluster in the cluster list
while clusters:
    # go through all the data points and add them to a cluster that they are the closest to
    for x in range(w):
        for y in range(h):
            if not clusters:
                break
            dist = np.inf
            for i in range(0, len(clusters)):
                euclidian = distance(clusters[i].mean, image[y, x])
                if euclidian < dist:
                    dist = euclidian
                    closest = i
            # alter cluster previous mean and current mean
            clusters[closest].prev = clusters[closest].mean.copy()
            clusters[closest].add(image[y, x])
    # if the cluster values aren't changing, put them in the finished list
    for j in range(0, len(clusters)):
        if j < len(clusters):
            if distance(clusters[j].prev, clusters[j].mean) < 10:
                finished.append(clusters.pop(j))

# change image colors to reflect closest cluster
for x in range(w):
    for y in range(h):
        dist = np.inf
        for i in range(0, len(finished)):
            euclidian = distance(finished[i].mean, image[y, x])
            if euclidian < dist:
                dist = euclidian
                closest = i
        image[y, x] = finished[closest].mean

cv2.imshow('Problem 4', image)
cv2.waitKey(0)








