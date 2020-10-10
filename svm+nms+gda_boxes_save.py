import numpy as np
import preprocess
#from keras.preprocessing import image
import cv2
from pathlib import Path
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
import pandas as pd
from builtins import range
from skimage.feature import hog
from skimage import exposure
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import os
# Ignore warnings
import warnings
import os
from skimage.feature import hog
from skimage import exposure
import math
from numpy import *
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pywt import dwt2
import pywt
import skimage
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from sklearn.mixture import GaussianMixture as GMM
from numpy.random import uniform, seed
from scipy.interpolate import griddata


warnings.filterwarnings('ignore')
print("Files imported successfully")
########################### GDA #####################################
def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    #print(size)
    assert (size == len(mu) and (size, size) == sigma.shape), "dims of input do not match"
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        assert det!=0, "covariance matrix cannot be singular"

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result


def load_images_from_folder_gda(folder):
    patches=[]
    fd_list=[]
    hog_img_list=[]
    for filename in os.listdir(folder):
        patch = cv2.imread(os.path.join(folder,filename))
        patch = cv2.resize(patch, (40, 40))
        #fd, hog_image = hog(patch, orientations=9, pixels_per_cell=(8, 8),
        #                        cells_per_block=(2, 2), visualize=True, multichannel=True)
        fd, hog_image = hog(patch, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
        hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        #cv2.imshow("hog", hog_image)
        #cv2.imshow("i", patch)
        #cv2.waitKey(0)
        if patch is not None:
            patches.append(patch)
            #fd = fd.reshape([fd.shape[0],1])
            fd_list.append(fd)
            hog_img_list.append(hog_image)
    return patches, np.array(fd_list), hog_img_list


def normalize(fd_list):
    mean = np.mean(fd_list, axis=0)
    sd = np.std(fd_list, axis=0)

    normal_fd = (fd_list - mean)/sd
    return normal_fd, mean, sd


def compute_covar(x_a, mu1):
    sub = x_a - mu1
    covar = np.zeros((mu.shape[0],mu.shape[0]))
    for i in sub:
        dot = np.dot(i.reshape(mu.shape[0],1), i.reshape(mu.shape[0],1).T)
        covar = covar + dot
    return covar/x_a.shape[0]


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


def add_new_features(patches):
    final_add=[]
    for patch in patches:
        spaces = []
        to_add = []
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

        final_add.append(np.array(to_add))
    final_add = np.array(final_add)
    norm_final_add, final_add_mean, final_add_sd = normalize(final_add)
    return norm_final_add, final_add_mean, final_add_sd


def give_prob(img, components, new_mean, new_sd, distribution_mu, distribution_covar):
    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, multichannel=True)
    pre_point = np.dot(components,fd)
    new_features = add_new_feature(img, new_mean, new_sd)
    test_features = np.append(pre_point, new_features)
    # print(test_features)
    # print(distribution_mu)
    # print(distribution_covar)
    # exit()
    prob = norm_pdf_multivariate(test_features, distribution_mu, matrix(distribution_covar))
    return prob



def gda_train(path):
    patches, fd_list, hog_img_list = load_images_from_folder_gda(path)
    print("\nShape of feature descriptor:",fd_list.shape)
    print("\nShape of each feature descriptor:", fd_list[0].shape)


    scaler = StandardScaler()
    scaler.fit(fd_list)
    fd_list = scaler.transform(fd_list) #fd_list is now normalized


    pca = PCA(n_components=30, svd_solver="full")
    #pca = PCA(0.95)    #You can find out how many components PCA choose after fitting the model using pca.n_components_
    pca.fit(fd_list)


    fd_list_pca = pca.transform(fd_list)
    ####################################################np.save("pca_components", pca.components_)


    print("\npca features shape before adding new features:",fd_list_pca.shape)
    print("sum of variance ratio:",np.sum(pca.explained_variance_ratio_))

    norm_new_add, new_add_mean, new_add_sd = add_new_features(patches)
    ###################################################np.save("new_add_mean_for_normalizing", new_add_mean)
    ####################################################np.save("new_add_sd_for_normalizing", new_add_sd)
    print("\nnew add features shape:", norm_new_add.shape)


    print("\n\nNow stacking pca and new added features together!!")
    final_features = np.column_stack((fd_list_pca, norm_new_add))
    print("final features shape is:",final_features.shape)



    print("\ntraining in process now!!\n")
    #computing mu and covariance matrix using maximum likelihood estimates
    mu = np.mean(final_features, axis=0)
    #covar = compute_covar(fd_list, mu)
    covar_np = np.cov(final_features, rowvar=False)

    return pca.components_, new_add_mean, new_add_sd, mu, covar_np



############################### SVM HOG NMS ########################

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




def non_max_suppression_slow(boxes, overlapThresh, w):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    # print(boxes)
    # print()
    # print(x1)
    # print()
    # print(y1)
    # print()
    # print(x2)
    # print()
    # print(y2)
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # print(area)
    # print()
    # print(idxs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        itsi = idxs[last]
        pick.append(itsi)
        suppress = [last]
        # loop over all indexes in the indexes list
        for pos in range(0, last):
            # grab the current index
            itsj = idxs[pos]
            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box1
            xx1 = x1[itsi] if x1[itsi] > x1[itsj] else x1[itsj]
            yy1 = y1[itsi] if y1[itsi] > y1[itsj] else y1[itsj]
            xx2 = x2[itsi] if x2[itsi] < x2[itsj] else x2[itsj]
            yy2 = y2[itsi] if y2[itsi] < y2[itsj] else y2[itsj]
            # compute the width and height of the bounding box
            ww = 0 if 0 > xx2 - xx1 + 1 else xx2 - xx1 + 1
            hh = 0 if 0 > yy2 - yy1 + 1 else yy2 - yy1 + 1
            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(ww * hh) / area[itsj]
            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlapThresh:
                suppress.append(pos)
        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)
    # return only the bounding boxes that were picked
    return boxes[pick]

def load_images_from_folder(folder):
    images=[]
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def load_image_names(folder):
    image_names=[]
    for filename in os.listdir(folder):
        f = os.path.splitext(filename)[0]
        image_names.append(f)
    return image_names

def eliminate_small(c):
    new = []
    for i in c:
        if(cv2.contourArea(i)<15):
            None
        else:
            new.append(i)
    return new

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

def find_contours(img):
    ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),127, 255, cv2.THRESH_BINARY)
    threshed_img = 255 - threshed_img
    _,contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def score(tp, fp, fn, tn):
    if(tp+fp!=0):
        precision = tp/(tp+fp)
    else:
        print("All are predicted as negative!(tp+fp)=0\n")
        precision = None

    if(tp+fn!=0):
        recall = tp/(tp+fn)
    else:
        print("There are no positive samples in input data!(tp+fn)=0\n")
        recall = None

    if(tn!=0):
        accuracy = (tp + tn)/(tp+tn+fp+fn)
    else:
        accuracy = None

    if(precision is None or recall is None):
        f1score = None
    else:
        f1score = (2*recall*precision)/(recall+precision)

    return [precision, recall, accuracy, f1score]

def find_segmented_contours(seg):
    _,contours, hier = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours



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

    '''# plotting contour
    plot_countour(xx, yy, z)
    plt.xlim(limclose[0], limfar[0])
    plt.ylim(limclose[1], limfar[1])
    plt.scatter(coordinates[:,0], coordinates[:,1])
    plt.show()
    '''

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


#############################################################################



w=40
h=40

#model = joblib.load('knot_model.pkl')
#model = joblib.load('more_data.pkl')
#model = joblib.load('multisize.pkl')

model = joblib.load('E:\\CVG\\MicroSuture\\knot_depth_estimation/files_for_svm/final/preprocess_data.pkl')
mu_for_norm = np.load('E:\\CVG\\MicroSuture\\knot_depth_estimation/files_for_svm/final/mu_for_norm.npy')
sd_for_norm = np.load('E:\\CVG\\MicroSuture\\knot_depth_estimation/files_for_svm/final/sd_for_norm.npy')


#model = joblib.load('pixels_per_cell_12.pkl')
# img = cv2.imread("F:\\files\\VCG\\AIIMS\\data\\newdata\\dataset\\test\\6.png")

#image_names_we_got = load_image_names("E:\\CVG\\MicroSuture\\knot_depth_estimation\\data2/")
#image_names_we_got = load_image_names("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_10_class/")
#image_names_we_got = load_image_names("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/")
image_names_we_got = load_image_names("E:\\CVG\\MicroSuture\\FINAL_DATA_SPLIT\\original_data\\test/")

print("images names are:", image_names_we_got)
print("\nnumber of images:", len(image_names_we_got))
print(image_names_we_got)
#exit()
for name in image_names_we_got:
    #img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\data2/"+name+".png")
    #img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_10_class/"+name+".png")
    #img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/"+name+".png")
    img = cv2.imread("E:\\CVG\\MicroSuture\\FINAL_DATA_SPLIT\\original_data\\test/"+name+".png")

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
        #print(original)
        #print(xsh, ysh)
        im_rescale = cv2.resize(new_img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
        imcopy = cv2.resize(img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)


        segmented = gmm_segmentation(im_rescale)
        #cv2.imshow("1", segmented)
        segmented = gda_on_image(segmented)

        #cv2.imshow("2", segmented)
        #cv2.imshow("img", img)
        #cv2.waitKey(0)

        #print(im_rescale.shape)

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


        # print("CUt 1 at:",cut1)
        # print("CUt 2 at:",cut2)



        patch = np.zeros((40,40,3),np.uint8)
        for width in range(5,im_rescale.shape[0]-40,10):
            for height in range(cut1,cut2-40,10):
                for i in range(40):
                    for j in range(40):
                        #print(i+width,j+height)
                        #patch[i][j] = img[i+height][j+width]
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
            cv2.rectangle(img, (iii[0],iii[1]), (round(iii[0]+w*(1/scale)),round(iii[1]+h*(1/scale))), (255,0,0), thickness=2)
            iii.append(iii[0]+round(w*(1/scale)))
            iii.append(iii[1]+round(h*(1/scale)))
            all_points.append(iii)
        start_points=[]

    all_boxes, all_boxes_probs = non_max_suppression(np.array(all_points), probs_list, 0.1) #0.1 for 5,7,8
                                                                                            #0.3 for class 10
    #all_boxes = non_max_suppression_slow(np.array(all_points), 0.3, w)

    ### TO DRAW BOUNDING BOXES AFTER NMS OPERATION
    for box in all_boxes:
        cv2.rectangle(img_sup, (box[0],box[1]), (box[2],box[3]), (255,0,0), thickness=2)



    # cv2.imshow("final", img)
    # cv2.imshow("final_supression", img_sup)

    # for box in all_boxes:
    #     cv2.imshow("dd",img[box[1]:box[1]+h,box[0]:box[0]+w])
    #     cv2.waitKey(0)
    print("SVM DONE and NMS done")
    # cv2.imshow("final", img)
    # cv2.imshow("final_supression", img_sup)
    #############################################################################################################

    # k_components, k_new_mean, k_new_sd, k_mu, k_covar = gda_train("F:/files/VCG/AIIMS/data/gdasvmhog/knot_dataset/old/new/train/train_knot")
    # nk_components, nk_new_mean, nk_new_sd, nk_mu, nk_covar = gda_train("F:/files/VCG/AIIMS/data/gdasvmhog/knot_dataset/old/new/train/train_nonknot")


    k_components = np.load("./final_files/k_components1200.npy")
    k_new_mean = np.load("./final_files/k_new_mean1200.npy")
    k_new_sd = np.load("./final_files/k_new_sd1200.npy")
    k_mu = np.load("./final_files/k_mu1200.npy")
    k_covar = np.load("./final_files/k_covar1200.npy")
    nk_components = np.load("./final_files/nk_components1200.npy")
    nk_new_mean = np.load("./final_files/nk_new_mean1200.npy")
    nk_new_sd = np.load("./final_files/nk_new_sd1200.npy")
    nk_mu = np.load("./final_files/nk_mu1200.npy")
    nk_covar = np.load("./final_files/nk_covar1200.npy")

    new_boxes=[]
    new_boxes_probs=[]
    for index,box in enumerate(all_boxes):
        i = im_rescale[box[1]:box[3],box[0]:box[2]]
        i = cv2.resize(i, (w, h), interpolation=cv2.INTER_CUBIC)
        #print(i.shape)
        #cv2.imshow("patch",i)
        #cv2.waitKey(0)
        knot_prob = give_prob(i, k_components, k_new_mean, k_new_sd, k_mu, k_covar)
        non_knot_prob = give_prob(i, nk_components, nk_new_mean, nk_new_sd, nk_mu, nk_covar)
        # print(knot_prob,non_knot_prob)
        if(knot_prob > non_knot_prob):
            new_boxes.append(box)
            new_boxes_probs.append(all_boxes_probs[index])

    #cv.imshow("contours", imcopy)
    print("\nGDA done!")

    for box in new_boxes:
        cv2.rectangle(backup, (box[0],box[1]), (box[2],box[3]), (255,0,0), thickness=2)

    #cv2.imshow("finallyyy",backup)
    #cv2.waitKey(0)

    print("\nsaving "+name+"'s boxes now!\n")

    #saving svm boxes in .txt
    #print(all_points)
    #print()
    #print(probs_list)
    file_svm = open("./final_predictions/svm/"+name+".txt", "w")
    for number in range(len(all_points)):
        xmin = str(all_points[number][0])
        ymin = str(all_points[number][1])
        xmax = str(all_points[number][2])
        ymax = str(all_points[number][3])
        confi = str(probs_list[number])
        class_is = "knot"
        L = [class_is+str(" "), confi+str(" "), xmin+str(" "), ymin+str(" "), xmax+str(" "), ymax+str("\n")]
        file_svm.writelines(L)
    file_svm.close()

    #saving nms boxes
    #print(all_boxes)
    #print()
    #print(all_boxes_probs)
    file_nms = open("./final_predictions/nms/"+name+".txt", "w")
    for number in range(len(all_boxes)):
        xmin = str(all_boxes[number][0])
        ymin = str(all_boxes[number][1])
        xmax = str(all_boxes[number][2])
        ymax = str(all_boxes[number][3])
        confi = str(all_boxes_probs[number])
        class_is = "knot"
        L = [class_is+str(" "), confi+str(" "), xmin+str(" "), ymin+str(" "), xmax+str(" "), ymax+str("\n")]
        file_nms.writelines(L)
    file_nms.close()

    #saving gda boxes
    #print(new_boxes)
    #print()
    #print(new_boxes_probs)
    file_gda = open("./final_predictions/gda/"+name+".txt", "w")
    for number in range(len(new_boxes)):
        xmin = str(new_boxes[number][0])
        ymin = str(new_boxes[number][1])
        xmax = str(new_boxes[number][2])
        ymax = str(new_boxes[number][3])
        confi = str(new_boxes_probs[number])
        class_is = "knot"
        L = [class_is+str(" "), confi+str(" "), xmin+str(" "), ymin+str(" "), xmax+str(" "), ymax+str("\n")]
        file_gda.writelines(L)
    file_gda.close()

    all_points = []
    probs_list = []
    all_boxes = []
    all_boxes_probs = []
    new_boxes = []
    new_boxes_probs = []




    #saving boxes for visualization
    cv2.imwrite("./final_image_results/"+name+"svm.png",img)
    cv2.imwrite("./final_image_results/"+name+"nms.png",img_sup)
    cv2.imwrite("./final_image_results/"+name+"gda.png",backup)

#cv2.imshow("./multisize/i_svm.png",img)
#cv2.imshow("./multisize/i_nms.png",img_sup)
#cv2.imshow("./multisize/i_gda.png",backup)
#cv2.waitKey(0)

