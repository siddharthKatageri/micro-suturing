import cv2
import numpy as np
import os
import preprocess
from sklearn.mixture import GaussianMixture as GMM
from skimage import morphology
from skimage import measure
from numpy import *

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
    #CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    CS = plt.contourf(xi,yi,zi,len(levels),cmap=cm.Greys_r, levels=levels)
    # plot data points.
    # plt.scatter(x, y, marker='o', c='b', s=5)
    #plt.show()


def gauss(x,y,Sigma,mu):
    X=np.vstack((x,y)).T
    mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
    return  np.diag(np.exp(-1*(mat_multi)))

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

def normalize(fd_list):
    mean = np.mean(fd_list, axis=0)
    sd = np.std(fd_list, axis=0)

    normal_fd = (fd_list - mean)/sd
    return normal_fd, mean, sd










def gda_on_image(image):
    #print(image.shape)
    find = np.where(image==[255])
    x = np.reshape(find[1],(-1 ,1))
    y = np.reshape(find[0],(-1 ,1))
    coordinates = np.hstack((x, y))
    coordinates, ori_mean, ori_std = normalize(coordinates)
    coordinates = np.array(coordinates)


    mu = np.mean(coordinates, axis=0)
    covar_np = np.cov(coordinates, rowvar=False)
    #print(mu)
    #print(covar_np)

    npts = 3000
    limclose = ([0, 0] - ori_mean)/ori_std
    limfar = (image.shape - ori_mean)/ori_std
    xx = uniform(-2, 2, npts)
    yy = uniform(-2, 2, npts)
    z = gauss(xx, yy, Sigma=covar_np, mu=mu)
    plot_countour(xx, yy, z)
    #plt.scatter(x, y)
    #plt.xlim(0, image.shape[1])
    #plt.ylim(0, image.shape[0])
    plt.xlim(limclose[0], limfar[0])
    plt.ylim(limclose[1], limfar[1])
    plt.scatter(coordinates[:,0], coordinates[:,1])
    plt.show()
    #prob = norm_pdf_multivariate(np.array([454.44537648,351.1505526]), mu, matrix(covar_np))

    r = np.abs(coordinates - mu)
    out = np.divide(r,[covar_np[0,0], covar_np[1,1]])
    #print(out)

    mask = out>2
    #print(mask)
    indices = [i for i,x in enumerate(mask) if True in x]
    #print()
    #print(indices)
    #print()

    #print(coordinates.shape)
    new_coordinates = np.delete(coordinates, indices, 0)
    #print(new_coordinates.shape)

    new_coordinates = np.add(np.multiply(new_coordinates, ori_std), ori_mean)
    new_coordinates = new_coordinates.astype('int')

    denoise = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    s = (new_coordinates[:,1]), (new_coordinates[:,0])
    #print(s)
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

w=40
h=40

#model = joblib.load('knot_model.pkl')
#model = joblib.load('more_data.pkl')
#model = joblib.load('multisize.pkl')

name = "73.png"
img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/"+name)

il, ir = preprocess.find_width(img)
new_img = preprocess.vectorize_strech(img, 200, 255, il, ir)

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

    im_rescale = cv2.resize(new_img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)
    imcopy = cv2.resize(img,(int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_CUBIC)

    segmented = gmm_segmentation(im_rescale)
    cv2.imshow("1", segmented)
    segmented = gda_on_image(segmented)

    cv2.imshow("2", segmented)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    exit()






    # -------------------
    contours = find_segmented_contours(segmented)
    #new = eliminate_small(contours)
    cv2.drawContours(imcopy, contours, -1, (255, 0, 0), 4)
    cv2.imshow("im", imcopy)
    cv2.imshow("seg", segmented)
    cv2.waitKey(0)

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





















