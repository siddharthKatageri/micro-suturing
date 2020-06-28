import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import cv2
from scipy.signal import find_peaks
from scipy.signal import peak_widths
from sklearn.mixture import GaussianMixture as GMM


'''
img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\data/1.png")

#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
vec = img.reshape((-1, 3))

# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                            random_state=0)

X = StandardScaler().fit_transform(X)
# Compute DBSCAN
db = DBSCAN(eps=0.3, min_samples=10).fit(vec)
print(dir(db))
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)


bin = center2[labels]
bin_image = bin.reshape((img.shape))
cv2.imshow("img", img)
#cv2.imshow("seg", res_image)
cv2.imshow("seg_bin", bin_image)
cv2.waitKey(0)
'''
def rescale(pin, a, b, c, d):
    f = pin - c
    s = (b-c)/(d-c)
    pout = (f*s)+a
    pout[pout<0] = 0
    pout[pout>255] = 255
    return np.round(pout)

def vectorize_strech(img, a, b, c, d):          # very fast than custom stretch
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img)

    sss = np.nonzero((y > c) & (y < d))
    y[sss] = rescale(y[sss], a, b, c, d)

    ddd = np.nonzero((y < c) & (y > 0))
    y[ddd] = rescale(y[ddd], 0, 20, 0, c)

    img = cv2.merge((y, cr, cb))
    new = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return new

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

def normalize(fd_list):
    mean = np.mean(fd_list, axis=0)
    sd = np.std(fd_list, axis=0)

    normal_fd = (fd_list - mean)/sd
    return normal_fd, mean, sd






def dbscan_on_image(image):
    find = np.where(image==[255])
    x = np.reshape(find[1],(-1 ,1))
    y = np.reshape(find[0],(-1 ,1))
    coordinates = np.hstack((x, y))
    coordinates, ori_mean, ori_std = normalize(coordinates)
    coordinates = np.array(coordinates)
    print(coordinates.shape)

    print("dbscan now!")
    db = DBSCAN(eps=2, min_samples=20, metric = 'euclidean',algorithm ='kd_tree', n_jobs=-1).fit(coordinates)
    print(db.labels_)
    exit()






name = "69.png"
img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\dataset_80_sutures/"+name)

il, ir = find_width(img)
new_img = vectorize_strech(img, 200, 255, il, ir)

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
    segmented = dbscan_on_image(segmented)

    cv2.imshow("2", segmented)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    exit()
