import cv2
import numpy as np
import os
import preprocess
from sklearn.mixture import GaussianMixture as GMM
from skimage import morphology

def gmm_segmentation(img):
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    vec = img.reshape((-1,3))
    gmm_model = GMM(n_components = 2,covariance_type = 'tied', n_init=5).fit(vec)
    gmm_labels = gmm_model.predict(vec)
    segmented = gmm_labels.reshape(img.shape[0],img.shape[1])
    segmented = np.multiply(segmented, 255)
    segmented = segmented.astype('uint8')
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #new_segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)
    new_segmented = morphology.remove_small_objects(segmented, min_size=100, connectivity=2)
    cv2.imshow("seg", segmented)
    cv2.imshow("new", new_segmented)
    cv2.waitKey(0)
    exit()
    return new_segmented

def find_segmented_contours(seg):
    _,contours, hier = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

w=40
h=40

#model = joblib.load('knot_model.pkl')
#model = joblib.load('more_data.pkl')
#model = joblib.load('multisize.pkl')

name = "80.png"
img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/"+name)

il, ir = preprocess.find_width(img)
new_img = preprocess.custom_strech(img, 200, 255, il, ir)

img_sup = img.copy()
backup = img.copy()
start_points = []
probs_list = []


for scale in [1]:
#for scale in [1, 0.6, 1.4]:
    ss = []
    #print("Scale:", scale)
    xsh = int(img.shape[1]*scale)
    ysh = int(img.shape[0]*scale)
    #print(original)
    #print(xsh, ysh)
    im_rescale = cv2.resize(new_img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
    imcopy = cv2.resize(img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)

    segmented = gmm_segmentation(im_rescale)

    contours = find_segmented_contours(segmented)
    #new = eliminate_small(contours)
    cv2.drawContours(imcopy, contours, -1, (255, 0, 0), 4)
    cv2.imshow("im", imcopy)
    cv2.imshow("seg", segmented)
    cv2.waitKey(0)
    exit()

    min, max = restrict(new)
    dis = max - min
    remain = im_rescale.shape[1] - max

    cut1 = min+dis//5
    cv2.line(imcopy, (min+dis//5,0), (min+dis//5,im_rescale.shape[0]), (0,0,255), 3)
    if(remain>5):
        cut2 = max+5
        cv2.line(imcopy, (max+5,0), (max+5,im_rescale.shape[0]), (0,0,255), 3)
    else:
        cut2 = im_rescale.shape[1]
        cv2.line(imcopy, (im_rescale.shape[1],0), (im_rescale.shape[1],im_rescale.shape[0]), (0,0,255), 3)


    # print("CUt 1 at:",cut1)
    # print("CUt 2 at:",cut2)





















