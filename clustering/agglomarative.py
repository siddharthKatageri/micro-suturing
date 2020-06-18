'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\data/1.png")
img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
vec = img.reshape((-1, 3))
vec = np.float32(vec)

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
cluster.fit_predict(vec)

print(cluster.label)


'''
import time as time
import cv2
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

###############################################################################
# Generate data
#lena = sp.misc.lena()
img = cv2.imread("E:\\CVG\\MicroSuture\\knot_depth_estimation\\data/1.png")
lena = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# Downsample the image by a factor of 4
lena = lena[::2, ::2] + lena[1::2, ::2] + lena[::2, 1::2] + lena[1::2, 1::2]
X = np.reshape(lena, (-1, 1))

###############################################################################
# Define the structure A of the data. Pixels connected to their neighbors.
connectivity = grid_to_graph(*lena.shape)

###############################################################################
# Compute clustering
print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 2  # number of regions
ward = AgglomerativeClustering(n_clusters=n_clusters,
        linkage='ward', connectivity=connectivity).fit(X)
label = np.reshape(ward.labels_, lena.shape)
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)

###############################################################################
# Plot the results on an image
plt.figure(figsize=(5, 5))
plt.imshow(lena, cmap=plt.cm.gray)
for l in range(n_clusters):
    plt.contour(label == l, contours=1,
                colors=[plt.cm.nipy_spectral(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
plt.show()

#cmap = cm.get_cmap("Spectral")
#colors = cmap(a / b)
