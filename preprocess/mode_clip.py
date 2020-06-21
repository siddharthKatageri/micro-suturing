import cv2
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from skimage import exposure


img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/8.png")
imgy = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
y, cr, cb = cv2.split(imgy)
yh = cv2.calcHist([y],[0],None,[256],[0,256])

#plt.plot(yh)
#plt.show()
y_rescale = exposure.rescale_intensity(y, in_range=(160, 190), out_range=(190,250))
y_rescale = y_rescale.astype('uint8')
yh_re = cv2.calcHist([y_rescale],[0],None,[256],[0,256])
plt.plot(yh_re)
plt.show()

new = cv2.merge((y_rescale, cr, cb))
new = cv2.cvtColor(new, cv2.COLOR_YCrCb2RGB)
cv2.imshow("new",new)
cv2.imshow("img", img)
cv2.waitKey(0)
