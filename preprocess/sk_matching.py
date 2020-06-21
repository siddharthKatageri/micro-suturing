import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import numpy as np
import os

def histogram_equalize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_b, img_g, img_r = cv2.split(img)
    #histr = cv2.calcHist([img],[0],None,[256],[0,256])
    img_r = cv2.equalizeHist(img_r)
    img_g = cv2.equalizeHist(img_g)
    img_b = cv2.equalizeHist(img_b)
    new = cv2.merge((img_r, img_g, img_b))
    return new

def match_wrt_channel(image, reference):
    ref_ycc = cv2.cvtColor(reference, cv2.COLOR_RGB2YCrCb)
    image_ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

    ref_y, ref_cr, ref_cb = cv2.split(ref_ycc)
    image_y, image_cr, image_cb = cv2.split(image_ycc)

    #histogram matching
    #matched = match_histograms(image, ref_e, multichannel=True)
    matched_y = match_histograms(image_y, ref_y, multichannel=False)
    matched_y = matched_y.astype('uint8')

    new = cv2.merge((matched_y, image_cr, image_cb))
    new = cv2.cvtColor(new, cv2.COLOR_YCrCb2RGB)
    return new


reference = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/6.png")
image = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/79.png")

new = match_wrt_channel(image, reference)

#direct = match_histograms(image, reference, multichannel=True)
cv2.imshow("Source", image)
cv2.imshow("reference", reference)
cv2.imshow("matched", new)
#cv2.imshow("direct", direct)
cv2.waitKey(0)

