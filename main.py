import cv2
import numpy as np
import os
import preprocess
from sklearn.mixture import GaussianMixture as GMM
from skimage import morphology
from skimage import measure
from numpy import *
from sklearn.externals import joblib
from skimage.feature import hog
from scipy.stats import multivariate_normal
from pywt import dwt2
import skimage


import matplotlib.pyplot as plt
from numpy.random import uniform, seed
from scipy.interpolate import griddata
from matplotlib import cm




def plot_countour(x,y,z):
    # define grid.
    xi = np.linspace(-2.1, 2.1, 100)
    yi = np.linspace(-2.1, 2.1, 100)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.1,0.2, 0.4, 0.6, 0.8, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt.contour(xi,yi,zi,len(levels),linewidths=0.5,colors='k', levels=levels)
    CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)


def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))


def normalize(fd_list):
    mean = np.mean(fd_list, axis=0)
    sd = np.std(fd_list, axis=0)

    normal_fd = (fd_list - mean)/sd
    return normal_fd, mean, sd


def gda_on_image(image):
    find = np.where(image==[255])
    x = np.reshape(find[1],(-1 ,1))
    y = np.reshape(find[0],(-1 ,1))
    coordinates = np.hstack((x, y))
    coordinates, ori_mean, ori_std = normalize(coordinates)
    coordinates = np.array(coordinates)


    mu = np.mean(coordinates, axis=0)
    covar_np = np.cov(coordinates, rowvar=False)

    npts = 3000
    limclose = ([0, 0] - ori_mean)/ori_std
    limfar = (image.shape - ori_mean)/ori_std
    xx = uniform(-2, 2, npts)
    yy = uniform(-2, 2, npts)
    z = gauss(xx, yy, Sigma=covar_np, mu=mu)

    # plotting contour
    #plot_countour(xx, yy, z)
    #plt.xlim(limclose[0], limfar[0])
    #plt.ylim(limclose[1], limfar[1])
    #plt.scatter(coordinates[:,0], coordinates[:,1])
    #plt.show()


    r = np.abs(coordinates - mu)
    out = np.divide(r,[covar_np[0,0], covar_np[1,1]])

    mask = out>2
    indices = [i for i,x in enumerate(mask) if True in x]

    new_coordinates = np.delete(coordinates, indices, 0)

    new_coordinates = np.add(np.multiply(new_coordinates, ori_std), ori_mean)
    new_coordinates = new_coordinates.astype('int')

    denoise = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    s = (new_coordinates[:,1]), (new_coordinates[:,0])
    denoise[s] = 255

    return denoise


def gmm_segmentation(img):
    #img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    vec = img.reshape((-1,3))
    gmm_model = GMM(n_components = 2,covariance_type = 'tied', n_init=5).fit(vec)
    gmm_labels = gmm_model.predict(vec)
    segmented = gmm_labels.reshape(img.shape[0],img.shape[1])
    segmented = np.multiply(segmented, 255)
    segmented = segmented.astype('uint8')
    whites = np.where(segmented==[255])
    blacks = np.where(segmented==[0])
    if(len(whites[0])>len(blacks[0])):
        segmented = np.subtract(255, segmented)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    new_segmented = cv2.morphologyEx(segmented, cv2.MORPH_OPEN, kernel)

    #cv2.imshow("seg", segmented)
    #cv2.imshow("new", new_segmented)
    #cv2.waitKey(0)
    return new_segmented


def find_segmented_contours(seg):
    _,contours, hier = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def restrict(new):
    min = 10000
    max = 0
    for i in new:
        for j in np.squeeze(i):
            if(j[0] < min):
                min = j[0]
            if(j[0]>max):
                max = j[0]
    return min, max

def eliminate_small(c):
    new = []
    for i in c:
        if(cv2.contourArea(i)<10):
            None
        else:
            new.append(i)
    return new

def load_images_from_folder(folder):
    images=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked

    return boxes[pick].astype("int"), np.array(probs)[pick]



def add_new_feature(patch, mu, sd):
    spaces=[]
    to_add=[]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    y = cv2.cvtColor(patch, cv2.COLOR_BGR2YCR_CB)
    spaces.append(patch)
    spaces.append(hsv)
    spaces.append(lab)
    spaces.append(y)
    for space in spaces:
        first = np.mean(np.reshape(space[:,:,0],-1))
        second = np.mean(np.reshape(space[:,:,1],-1))
        third = np.mean(np.reshape(space[:,:,2],-1))
        to_add.append(first)
        to_add.append(second)
        to_add.append(third)
    # # ENERGY
    im = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, (cH, cV, cD) = dwt2(im.T, 'db1')
    Energy = (cH**2 + cV**2 + cD**2).sum()/im.size
    to_add.append(Energy)

    # ENTROPY
    entropy = skimage.measure.shannon_entropy(patch)
    to_add.append(entropy)

    # CONTRAST
    Y = cv2.cvtColor(patch, cv2.COLOR_BGR2YUV)[:,:,0]
    min = np.int32(np.min(Y))
    max = np.int32(np.max(Y))
    if(min+max==0):
        contrast = 0
    else:
        contrast = (max-min)/(max+min)
    to_add.append(contrast)

    # HU MOMENTS
    # moments = cv2.moments(im)
    # huMoments = cv2.HuMoments(moments)
    # huMoments = np.ravel(huMoments)
    # for i in huMoments:
    #     to_add.append(i)

    to_add = np.array(to_add)
    to_add = (to_add-mu)/sd
    return to_add





def give_prob(img, components, new_mean, new_sd, distribution_mus, distribution_covars, weights):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    pre_point = np.dot(components,fd)
    new_features = add_new_feature(img, new_mean, new_sd)
    test_features = np.append(pre_point, new_features)
    test_features = np.expand_dims(test_features, axis=0)


    weights = weights.reshape(1,len(weights))
    gen_probs = np.zeros((len(test_features), len(distribution_mus)))
    for ind, l in enumerate(range(len(distribution_mus))):
        var = multivariate_normal(mean=distribution_mus[l], cov=distribution_covars[l])  #multivariate_normal is sklearn's implementation
        gen_probs[:,ind] = var.pdf(test_features)                                      # for my norm_pdf_multivariate function defined above
    for ele in gen_probs:
        #print("prob(x):\n", np.sum(np.multiply(gmm.weights_, ele)))
        prob = np.sum(np.multiply(weights, ele))

    return prob


def add_new_feature(patch, mu, sd):
    spaces=[]
    to_add=[]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    y = cv2.cvtColor(patch, cv2.COLOR_BGR2YCR_CB)
    spaces.append(patch)
    spaces.append(hsv)
    spaces.append(lab)
    spaces.append(y)
    for space in spaces:
        first = np.mean(np.reshape(space[:,:,0],-1))
        second = np.mean(np.reshape(space[:,:,1],-1))
        third = np.mean(np.reshape(space[:,:,2],-1))
        to_add.append(first)
        to_add.append(second)
        to_add.append(third)
    # ENERGY
    im = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, (cH, cV, cD) = dwt2(im.T, 'db1')
    Energy = (cH**2 + cV**2 + cD**2).sum()/im.size
    to_add.append(Energy)

    # ENTROPY
    entropy = skimage.measure.shannon_entropy(patch)
    to_add.append(entropy)

    # CONTRAST
    Y = cv2.cvtColor(patch, cv2.COLOR_BGR2YUV)[:,:,0]
    min = np.int32(np.min(Y))
    max = np.int32(np.max(Y))
    if(min+max==0):
        contrast = 0
    else:
        contrast = (max-min)/(max+min)
    to_add.append(contrast)

    to_add = np.array(to_add)
    to_add = (to_add-mu)/sd
    return to_add



w=40
h=40

#model = joblib.load('knot_model.pkl')
#model = joblib.load('more_data.pkl')
#model = joblib.load('multisize.pkl')

model = joblib.load('E:\\CVG\\MicroSuture\\knot_depth_estimation/files_for_svm/final/preprocess_data.pkl')
mu_for_norm = np.load('E:\\CVG\\MicroSuture\\knot_depth_estimation/files_for_svm/final/mu_for_norm.npy')
sd_for_norm = np.load('E:\\CVG\\MicroSuture\\knot_depth_estimation/files_for_svm/final/sd_for_norm.npy')

name = "4.png"
img = cv2.imread("E:\\CVG\\MicroSuture\\FINAL_DATA_SPLIT\\original_data\\test/"+name)

#images = load_images_from_folder("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/")
#for img in images:

il, ir = preprocess.find_width(img)
new_img = preprocess.vectorize_strech(img, 200, 255, il, ir)

img_sup = img.copy()
backup = img.copy()
start_points = []
probs_list = []
all_points=[]



for scale in [1]:
#for scale in [1, 0.6, 1.4]:
    ss = []
    #print("Scale:", scale)
    xsh = int(img.shape[1]*scale)
    ysh = int(img.shape[0]*scale)
                            #change img->new_img
    im_rescale = cv2.resize(new_img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)##marked
    imcopy = cv2.resize(img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
                                                                                                            #when using preprocessed data
    #new_img = cv2.resize(new_img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)##remove this line
    segmented = gmm_segmentation(im_rescale)#change new_img to im_rescale
    #cv2.imshow("1", segmented)
    segmented = gda_on_image(segmented)

    #cv2.imshow("2", segmented)
    #cv2.imshow("img", img)
    #cv2.imshow("newimg", new_img)
    #cv2.waitKey(0)


    # -------------------
    contours = find_segmented_contours(segmented)
    #new = eliminate_small(contours)
    cv2.drawContours(imcopy, contours, -1, (255, 0, 0), 4)


    min, max = restrict(contours)
    #dis = max - min
    remain = im_rescale.shape[1] - max

    #cut1 = min+dis//5
    cut1 = min
    cv2.line(imcopy, (cut1,0), (cut1,im_rescale.shape[0]), (0,0,255), 3)
    if(remain>10):
        cut2 = max+10
        cv2.line(imcopy, (cut2,0), (cut2,im_rescale.shape[0]), (0,0,255), 3)
    else:
        cut2 = im_rescale.shape[1]
        cv2.line(imcopy, (im_rescale.shape[1],0), (im_rescale.shape[1],im_rescale.shape[0]), (0,0,255), 3)



    patch = np.zeros((40,40,3),np.uint8)
    for width in range(5,im_rescale.shape[0]-40,10):    # earlier range(10, im_rescale.shape[0]-40,10)
        for height in range(cut1,cut2-40,10):
            for i in range(40):
                for j in range(40):
                    patch[i][j] = im_rescale[i+width][j+height]
            #cv2.imshow("p",patch)
            #cv2.waitKey(0)
            fd, hog_image = hog(patch, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True, multichannel=True)

            added_features = add_new_feature(patch, mu_for_norm, sd_for_norm)
            final_features = np.hstack((np.array(fd),added_features))
            final_features = final_features.reshape(final_features.shape[0],1)

            ans = model.predict(final_features.T)[0]
            p = model.predict_proba(final_features.T)
            if(ans == 1.0 and np.squeeze(p)[1]>0.68):
                probs_list.append(np.squeeze(p)[1])
                #cv.imshow("p",patch)
                #cv.waitKey(0)
                #cv.circle(imcopy,(height, width), 2, (255,255,0), thickness=2)
                #print(np.squeeze(p))
                #start_points.append([width,height])
                start_points.append([round(height*(1/scale)),round(width*(1/scale))])
                ss.append([height,width])

    '''
    for i in ss:
        cv2.rectangle(imcopy, (i[0],i[1]), (i[0]+w,i[1]+h), (255,0,0), thickness=1)
    #cv.imshow("contours", imcopy)
    #cv2.waitKey(0)
    '''

###################1.all_points is boxes after svm -> probs_list is the corresponding confidence
###################2.all_boxes is boxes after nms -> all_boxes_probs is the corresponding confidence
###################3.new_boxes is boxes after gda -> new_boxes_probs is the corresponding confidencw


    for iii in start_points:
        cv2.rectangle(img, (iii[0],iii[1]), (round(iii[0]+w*(1/scale)),round(iii[1]+h*(1/scale))), (255,0,0), thickness=1)
        iii.append(iii[0]+round(w*(1/scale)))
        iii.append(iii[1]+round(h*(1/scale)))
        all_points.append(iii)
    start_points=[]

all_boxes, all_boxes_probs = non_max_suppression(np.array(all_points), probs_list, 0.1) #0.1 for all classes other than class 10.
                                                                                        #0.3 for class 10
### TO DRAW BOUNDING BOXES AFTER NMS OPERATION
for box in all_boxes:
    cv2.rectangle(img_sup, (box[0],box[1]), (box[2],box[3]), (255,0,0), thickness=2)


print("SVM DONE and NMS done")
#####################################################################3
# GMM Classifier (commenting out gmm classifier and stoping the algorithm at nms itself.)
'''
k_components = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3k_components_1200.npy")
k_new_mean = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3k_new_mean_1200.npy")
k_new_sd = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3k_new_sd_1200.npy")
k_mus = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3k_mus_1200.npy")
k_covars = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3k_covars_1200.npy")
k_weights = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3k_weights_1200.npy")

nk_components = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3nk_components_1200.npy")
nk_new_mean = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3nk_new_mean_1200.npy",)
nk_new_sd = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3nk_new_sd_1200.npy")
nk_mus = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3nk_mus_1200.npy")
nk_covars = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3nk_covars_1200.npy")
nk_weights = np.load("E:/CVG/MicroSuture/GMM/files/final_files/3nk_weights_1200.npy")


new_boxes=[]
new_boxes_probs=[]
for index,box in enumerate(all_boxes):
    i = im_rescale[box[1]:box[3],box[0]:box[2]]
    i = cv2.resize(i, (w, h), interpolation=cv2.INTER_CUBIC)
    #print(i.shape)
    #cv2.imshow("patch",i)
    #cv2.waitKey(0)
    knot_prob = give_prob(i, k_components, k_new_mean, k_new_sd, k_mus, k_covars, k_weights)
    non_knot_prob = give_prob(i, nk_components, nk_new_mean, nk_new_sd, nk_mus, nk_covars, nk_weights)
    if(knot_prob > non_knot_prob):
        new_boxes.append(box)
        new_boxes_probs.append(all_boxes_probs[index])

print("\nGMM done!")

for box in new_boxes:
    cv2.rectangle(backup, (box[0],box[1]), (box[2],box[3]), (255,0,0), thickness=1)

'''
#cv2.imshow("gmm",backup)# this will output after gmm output
cv2.imshow("img",img)
cv2.imshow("sup",img_sup)
cv2.imshow("contours", imcopy)
cv2.waitKey(0)















