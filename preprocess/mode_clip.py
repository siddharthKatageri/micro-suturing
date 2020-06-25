import cv2
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from skimage import exposure
from scipy.signal import find_peaks
from scipy.signal import peak_widths

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


image = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/8.png")

il, ir = find_width(image)
new_image = custom_strech(image, 200, 255, il, ir)

cv2.imshow("image", image)
cv2.imshow("mc", new_image)
cv2.waitKey(0)
