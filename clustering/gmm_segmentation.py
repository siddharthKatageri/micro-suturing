import numpy as np
import cv2
import os
from sklearn.mixture import GaussianMixture as GMM
from scipy.stats import skew
import statistics
import time

img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/15.png")
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
vec = img.reshape((-1,3))
start_time = time.time()
gmm_model = GMM(n_components = 2,covariance_type = 'tied', n_init=1).fit(vec)
print("--- %s seconds ---" % (time.time() - start_time))
gmm_labels = gmm_model.predict(vec)
segmented = gmm_labels.reshape(img.shape[0],img.shape[1])
segmented = np.multiply(segmented, 255)
cv2.imshow("seg", np.uint8(segmented))
cv2.waitKey(0)

'''
folder="F:\\files\\VCG\\AIIMS\\data\\fulldata\\"
s=[]
# folder = "F:\\files\\VCG\\AIIMS\\data\\gdasvmhog\\1200\\Test\\test_nonknot"
for filename in sorted(os.listdir(folder)):
    img = cv2.imread(os.path.join(folder,filename))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # equ = cv2.equalizeHist(img)       ####histogram equlization
    img = cv2.GaussianBlur(img,(5,5),0)
    img2 = img.reshape((-1,3))
    gmm_model = GMM(n_components = 2,covariance_type = 'tied').fit(img2)
    gmm_labels = gmm_model.predict(img2)
    segmented = gmm_labels.reshape(img.shape[0],img.shape[1])
    s = segmented*255
    extn = os.path.split(filename)[1]
    cv2.imwrite("./gaussuanFitHSV5x5/"+str(extn)+".png",s)

print("done")
'''
