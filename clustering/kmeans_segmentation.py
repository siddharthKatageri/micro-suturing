import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\data/1.png")

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
vec = img.reshape((-1, 3))
vec = np.float32(vec)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

k = 2
attempts = 10
ret, label, center = cv2.kmeans(vec, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)

center2 = np.array([[255, 255, 255],
                    [0, 0, 0]], np.uint8)
bin = center2[label.flatten()]
bin_image = bin.reshape((img.shape))

#res = center[label.flatten()]
#res_image = res.reshape((img.shape))

cv2.imshow("img", img)
#cv2.imshow("seg", res_image)
cv2.imshow("seg_bin", bin_image)
cv2.waitKey(0)
