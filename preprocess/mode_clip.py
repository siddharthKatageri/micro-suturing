import cv2
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from skimage import exposure
from scipy.signal import find_peaks
from scipy.signal import peak_widths
import time
from itertools import product


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

def plot_y_hist(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    plt.hist(image[:,:,0].ravel(),256,[0,256])
    plt.show()

def rescale(pin, a, b, c, d):
    f = pin - c
    s = (b-c)/(d-c)
    pout = (f*s)+a
    pout[pout<0] = 0
    pout[pout>255] = 255
    return np.round(pout)


def custom_strech(img, a, b, c, d):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j][0]>c and img[i][j][0]<d):
                out = rescale(img[i][j][0], a, b, c, d)
                img[i][j][0] = out
            if(img[i][j][0]<c and img[i][j][0]>0):
                out = rescale(img[i][j][0], 0, 20, 0, c)
                img[i][j][0] = out
    new = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    return new

def custom_strech_fast(img, a, b, c, d):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            pixel = img.item(i, j, 0)
            if(pixel>c and pixel<d):
                out = rescale(pixel, a, b, c, d)
                img[i][j][0] = out
            if(pixel<c and pixel>0):
                out = rescale(pixel, 0, 20, 0, c)
                img[i][j][0] = out
    new = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return new

def custom_strech_ffast(img, a, b, c, d):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img)
    h, w = y.shape
    for pos in product(range(h), range(w)):
        pixel = y.item(pos)
        if(pixel>c and pixel<d):
            out = rescale(pixel, a, b, c, d)
            y[pos] = out
        if(pixel<c and pixel>0):
            out = rescale(pixel, 0, 20, 0, c)
            y[pos] = out
    img = cv2.merge((y, cr, cb))
    new = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return new



def vectorize_strech(img, a, b, c, d):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img)

    sss = np.nonzero((y > c) & (y < d))
    y[sss] = rescale(y[sss], a, b, c, d)

    ddd = np.nonzero((y < c) & (y > 0))
    y[ddd] = rescale(y[ddd], 0, 20, 0, c)

    img = cv2.merge((y, cr, cb))
    new = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return new






image = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/81.png")
print(image.shape)

il, ir = find_width(image)

start_time = time.time()
vec = vectorize_strech(image, 200, 255, il, ir)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
new_image = custom_strech(image, 200, 255, il, ir)
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
new_image_fast = custom_strech_fast(image, 200, 255, il, ir)
print("--- %s seconds ---" % (time.time() - start_time))
#plot_y_hist(new_image)

start_time = time.time()
new_image_ffast = custom_strech_ffast(image, 200, 255, il, ir)
print("--- %s seconds ---" % (time.time() - start_time))

cv2.imshow("image", image)
cv2.imshow("mc", new_image)
cv2.imshow("mcc", new_image_fast)
cv2.imshow("mccc", new_image_ffast)
cv2.imshow("vec", vec)
cv2.waitKey(0)
