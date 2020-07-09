import numpy as np
import cv2

from sklearn.mixture import GaussianMixture as GMM
import preprocess
from numpy.random import uniform, seed
import os

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

def eliminate_small(c):
    new = []
    for i in c:
        if(cv2.contourArea(i)<10):
            None
        else:
            new.append(i)
    return new

def find_segmented_contours(seg):
    _,contours, hier = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def eliminate_small(c):
    new = []
    for i in c:
        if(cv2.contourArea(i)<15):
            None
        else:
            new.append(i)
    return new

def load_image_names(folder):
    image_names=[]
    for filename in os.listdir(folder):
        f = os.path.splitext(filename)[0]
        image_names.append(f)
    return image_names




def morph(seg, tenth):
    if(tenth==True):
        length = 50

    else:
        length = 25

    print("length is:", length)
    #closing for filling null space in between(Note: Should perfrom Closing with kernel (50,3) for class 10 images)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(length,3))
    closing = cv2.morphologyEx(segmented_denoise, cv2.MORPH_CLOSE, kernel)

    if(not tenth):
        print("inside opening")
        #opening for elimination small connected components(Note: SHould not perform on class 10 images)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    return closing





name = "6.png"
img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/"+name)
#img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_10_class/"+name)
print(img.shape)

il, ir = preprocess.find_width(img)
new_img = preprocess.vectorize_strech(img, 200, 255, il, ir)

segmented = gmm_segmentation(new_img)
segmented_denoise = gda_on_image(segmented)

closed = morph(segmented_denoise, False)

#closing_contours = find_segmented_contours(closing)
#new_closing_contours = eliminate_small(closing_contours)
#cv2.drawContours(img, new_closing_contours, -1, (255, 0, 0), 4)


cv2.imshow("img", img)
cv2.imshow("denoise", segmented_denoise)
cv2.imshow("close", closed)
cv2.waitKey(0)

