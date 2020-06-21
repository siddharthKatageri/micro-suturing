import matplotlib.pyplot as plt
import numpy as np
from skimage import data, img_as_float
from skimage import exposure
import cv2
import os

img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/2.png")
# Contrast stretching
p5 = np.percentile(img, 5)
print(p5)
p95 = np.percentile(img, 95)
print(p95)
img_rescale = exposure.rescale_intensity(img, in_range=(p5, p95), out_range=(0,255))#, out_range=(0,255)
img8 = img_rescale.astype('uint8')
cv2.imshow("strech",img8)
cv2.imshow("original",img)
cv2.waitKey(0)
'''
folder="E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures"
for filename in sorted(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder,filename))

    p5 = np.percentile(img, 5)
    p95 = np.percentile(img, 95)
    img_rescale = exposure.rescale_intensity(img, in_range=(p5, p95), out_range=(0,255))#, out_range=(0,255)
    img8 = img_rescale.astype('uint8')
    extn = os.path.split(filename)[1]
    cv2.imwrite("./output/contrast/"+str(extn)+".png",img8)

print("done")
'''
