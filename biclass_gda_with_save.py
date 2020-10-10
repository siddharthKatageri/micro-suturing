#=COUNTIF(B2:XZ650,">0.9")
import numpy as np
import cv2
import os
import skimage
from skimage.feature import hog
from skimage import exposure
import math
from numpy import *
import seaborn as sn
import glob
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pywt import dwt2
import pywt


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


def load_image_files(container_path, block, dimension=(40, 40)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    print(folders)

    categories = [fo.name for fo in folders]
    descr = "A image classification dataset"
    images = []
    hog_images = []
    flat_data = []
    target = []
    count = 0
    train_fd = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            count += 1
            file = str(file)
            img = imread(file)
            img= cv.resize(img,(40,40))
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            images.append(img)
            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(block, block),
                                cells_per_block=(2, 2), visualize=True, multichannel=True)
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            train_fd.append(fd)
            hog_images.append(hog_image_rescaled)
    X = np.array(train_fd)
    # np.savetxt("only_40x40_new.txt",train_fd[0])
    return X,images


def load_images_from_folder(folder):
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
    min = np.int32(np.min(Y))         #######################################Change here.... added int32
    max = np.int32(np.max(Y))
    if(min+max==0):
        contrast = 0
    else:
        contrast = (max-min)/(max+min)
    to_add.append(contrast)

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
        min = np.int32(np.min(Y))   #######################################Change here.... added int32
        max = np.int32(np.max(Y))
        if(min+max==0):
            contrast = 0
        else:
            contrast = (max-min)/(max+min)
        to_add.append(contrast)

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
    patches, fd_list, hog_img_list = load_images_from_folder(path)
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
    Nans = isnan(covar_np)
    covar_np[Nans] = 0
    return pca.components_, new_add_mean, new_add_sd, mu, covar_np






k_components, k_new_mean, k_new_sd, k_mu, k_covar = gda_train("E:\\CVG\\MicroSuture\\FINAL_DATASET\\train/knot_patches_extracted")
nk_components, nk_new_mean, nk_new_sd, nk_mu, nk_covar = gda_train("E:\\CVG\\MicroSuture\\FINAL_DATASET\\train/nonknot_patches_extracted")

'''
np.save("./final_files/k_components1200_nc.npy",k_components)
np.save("./final_files/k_new_mean1200_nc.npy",k_new_mean)
np.save("./final_files/k_new_sd1200_nc.npy",k_new_sd)
np.save("./final_files/k_mu1200_nc.npy",k_mu)
np.save("./final_files/k_covar1200_nc.npy",k_covar)

np.save("./final_files/nk_components1200_nc.npy",nk_components)
np.save("./final_files/nk_new_mean1200_nc.npy",nk_new_mean)
np.save("./final_files/nk_new_sd1200_nc.npy",nk_new_sd)
np.save("./final_files/nk_mu1200_nc.npy",nk_mu)
np.save("./final_files/nk_covar1200_nc.npy",nk_covar)
'''

p_k =0
p_nk = 0
images = glob.glob("E:\\CVG\\MicroSuture\\FINAL_DATASET\\test/knot_patches_extracted/*.png")
for i in images:
    # img = cv2.imread("F:/files/VCG/AIIMS/data/gdasvmhog/knot_dataset/new/nonknot/30.png")
    img = cv2.imread(i)
    knot_prob = give_prob(img, k_components, k_new_mean, k_new_sd, k_mu, k_covar)
    non_knot_prob = give_prob(img, nk_components, nk_new_mean, nk_new_sd, nk_mu, nk_covar)
    if(knot_prob > non_knot_prob):
        # print("its knot",count)
        p_k+=1
    else:
        # print("its non knot",count)
        p_nk+=1

print(p_k)
print(p_nk)
print(p_k/(p_k+p_nk))


print("")

p_k =0
p_nk = 0
images = glob.glob("E:\\CVG\\MicroSuture\\FINAL_DATASET\\test/nonknot_patches_extracted/*.png")
for i in images:
    # img = cv2.imread("F:/files/VCG/AIIMS/data/gdasvmhog/knot_dataset/new/nonknot/30.png")
    img = cv2.imread(i)
    knot_prob = give_prob(img, k_components, k_new_mean, k_new_sd, k_mu, k_covar)
    non_knot_prob = give_prob(img, nk_components, nk_new_mean, nk_new_sd, nk_mu, nk_covar)
    if(knot_prob > non_knot_prob):
        # print("its knot",count)
        p_k+=1
    else:
        # print("its non knot",count)
        p_nk+=1

print(p_k)
print(p_nk)
print(p_nk/(p_k+p_nk))




# print("knot probability:",knot_prob)
# print("NOn knot probability:",non_knot_prob)

'''
prob = norm_pdf_multivariate(final_features[0], mu, matrix(covar_np))
print("\nprobability:", prob)
np.save("mu_pca", mu)
np.save("covar_pca", covar_np)
'''
