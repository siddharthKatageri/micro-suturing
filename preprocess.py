import matplotlib.pyplot as plt
import cv2
from skimage import data
from skimage import exposure
from skimage.exposure import match_histograms
import numpy as np
import os
from scipy.signal import argrelextrema
from scipy.signal import find_peaks
from scipy.signal import peak_widths

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
    return np.round(pout)


def custom_strech(img, a, b, c, d):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j][0]>c and img[i][j][0]<d):
                out = rescale(img[i][j][0], a, b, c, d)
                img[i][j][0] = out
    new = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    return new

def find_width(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    hist,bins = np.histogram(image[:,:,0].ravel(),256,[0,256])

    peaks, _ = find_peaks(hist)
    max_peak=max(hist[peaks])
    #to find the intensity of the pixel with highest frequency
    for i in range(len(peaks)):
        if hist[peaks[i]]==max_peak:
            inst=peaks[i]
            break

    result=peak_widths(hist,[inst], rel_height=0.85)
    result = list(result)
    result[2]=result[2]-15
    result[3]=result[3]+10
    #plt.plot(hist)
    #plt.plot(inst, hist[inst], "x")
    #plt.hlines(*result[1:], color="C3")
    #plt.show()
    return np.round(result[2]), np.round(result[3])

'''
reference = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/6.png")
image = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/1.png")



#mode clipping -> histogram specification
rl, rr = find_width(reference)
new_ref = custom_strech(reference, 200, 255, rl, rr)
#plot_y_hist(new_ref)


il, ir = find_width(image)
new_image = custom_strech(image, 200, 255, il, ir)
#plot_y_hist(new_image)


matched = match_wrt_channel(new_image, new_ref)
#plot_y_hist(matched)


cv2.imshow("new_ref", new_ref)
cv2.imshow("new_image", new_image)
cv2.imshow("reference", reference)
cv2.imshow("image", image)
cv2.imshow("matched", matched)
cv2.waitKey(0)
'''





#histtogram specification -> mode clipping
'''
plot_y_hist(image)
matched = match_wrt_channel(image, reference)
plot_y_hist(matched)
inst = find_maxpeak(matched)



#streched = strech_wrt_channel(matched)
#plot_y_hist(streched)
#cv2.imshow("strech",streched)
#cv2.waitKey(0)


new = custom_strech(matched, 200, 255, inst-50, inst+40)
plot_y_hist(new)
cv2.imshow("Source", image)
cv2.imshow("reference", reference)
cv2.imshow("matched", matched)
cv2.imshow("new", new)
cv2.waitKey(0)
'''

























#to apply pre-processing for all images in a folder
'''
reference = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/6.png")
folder="E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures"
for filename in sorted(os.listdir(folder)):
    image = cv2.imread(os.path.join(folder,filename))
    matched = match_wrt_channel(image, reference)

    inst = find_maxpeak(matched)
    new = custom_strech(matched, 200, 255, inst-50, inst+40)

    extn = os.path.split(filename)[1]
    cv2.imwrite("./output/specification/"+str(extn),matched)
    cv2.imwrite("./output/specification+modeclip/"+str(extn),new)

print("done")
'''
