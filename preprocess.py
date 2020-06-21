import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import numpy as np
import os
from scipy.signal import argrelextrema
from scipy.signal import find_peaks


def find_maxpeak(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    hist,bins = np.histogram(image[:,:,0].ravel(),256,[0,256])

    peaks, _ = find_peaks(hist)
    max_peak=max(hist[peaks])
    #to find the intensity of the pixel with highest frequency
    for i in range(len(peaks)):
        if hist[peaks[i]]==max_peak:
            inst=peaks[i]
            break
    return inst


def plot_y_hist(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    plt.hist(image[:,:,0].ravel(),256,[0,256])
    plt.show()

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

def strech_wrt_channel(image):
    image_ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    image_y, image_cr, image_cb = cv2.split(image_ycc)

    p5 = np.percentile(image_y, 5)
    p95 = np.percentile(image_y, 95)
    image_y_rescale = exposure.rescale_intensity(image_y, in_range=(p5, p95), out_range=(0,255))#, out_range=(0,255)
    image_y_rescale = image_y_rescale.astype('uint8')

    new = cv2.merge((image_y_rescale, image_cr, image_cb))
    new = cv2.cvtColor(new, cv2.COLOR_YCrCb2RGB)
    return new


def rescale(pin, a, b, c, d):
    f = pin - c
    s = (b-c)/(d-c)
    pout = (f*s)+a
    if(pout<0):
        return 0
    if(pout>255):
        return 255
    return round(pout)


def custom_strech(img, a, b, c, d):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j][0]>c and img[i][j][0]<d):
                out = rescale(img[i][j][0], a, b, c, d)
                img[i][j][0] = out
    new = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    return new




reference = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/6.png")
image = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/2.png")
plot_y_hist(image)

matched = match_wrt_channel(image, reference)
plot_y_hist(matched)
inst = find_maxpeak(matched)

cv2.imshow("Source", image)
cv2.imshow("reference", reference)
cv2.imshow("matched", matched)

streched = strech_wrt_channel(matched)
plot_y_hist(streched)
cv2.imshow("strech",streched)
cv2.waitKey(0)


new = custom_strech(matched, 200, 255, inst-40, inst+40)
plot_y_hist(new)


cv2.imshow("new", new)
cv2.waitKey(0)



























'''
reference = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/6.png")
folder="E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures"
for filename in sorted(os.listdir(folder)):
    image = cv2.imread(os.path.join(folder,filename))
    #reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #histogram equilization
    #ref_e = histogram_equalize(reference)



    #histogram matching
    #matched = match_histograms(image, ref_e, multichannel=True)
    matched = match_wrt_channel(image, reference)

    streched = strech_wrt_channel(matched)

    extn = os.path.split(filename)[1]
    cv2.imwrite("./output/spec+contrast/"+str(extn),streched)

print("done")
'''
