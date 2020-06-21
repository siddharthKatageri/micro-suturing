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

def strech(image):
    p5 = np.percentile(image, 5)
    p95 = np.percentile(image, 95)
    image_rescale = exposure.rescale_intensity(image, in_range=(p5, p95), out_range=(0,255))#, out_range=(0,255)
    image_rescale = image_rescale.astype('uint8')
    return image_rescale


def mode_clip_wrt_channel(image):
    image_ycc = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    image_y, image_cr, image_cb = cv2.split(image_ycc)
    image_y_hist = cv2.calcHist([image_y],[0],None,[256],[0,256])

    plt.plot(image_y_hist)
    plt.show()



reference = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/6.png")
image = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/1.png")

matched = match_wrt_channel(image, reference)

#direct = match_histograms(image, reference, multichannel=True)
cv2.imshow("Source", image)
cv2.imshow("reference", reference)
cv2.imshow("matched", matched)
#cv2.imshow("direct", direct)
#cv2.waitKey(0)

streched = strech_wrt_channel(matched)
cv2.imshow("strech",streched)
cv2.waitKey(0)

#mode_clip_wrt_channel(image)
#mode_clip_wrt_channel(matched)
#mode_clip_wrt_channel(streched)



























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
