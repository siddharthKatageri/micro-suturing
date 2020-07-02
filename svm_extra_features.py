import numpy as np
#from keras.preprocessing import image
import cv2 as cv
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
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.model_selection import KFold
from pywt import dwt2
import pywt
import skimage
# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
print("Files imported successfully")

def normalize(fd_list):
    mean = np.mean(fd_list, axis=0)
    sd = np.std(fd_list, axis=0)

    normal_fd = (fd_list - mean)/sd
    return normal_fd, mean, sd

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


        final_add.append(np.array(to_add))
    final_add = np.array(final_add)
    norm_final_add, final_add_mean, final_add_sd = normalize(final_add)
    return norm_final_add




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
            img = cv.imread(file)
            #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            #img = cv.resize(img,(40,40))
            images.append(img)
            fd, hog_image = hog(img, orientations=9, pixels_per_cell=(block, block),
                                cells_per_block=(2, 2), visualize=True, multichannel=True)
            if(len(fd)!=576):
                print(file)
                cv.imshow("i",img)
                cv.waitKey(0)
            assert len(fd) ==  576, "fd not of size 576"
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            train_fd.append(fd)
            hog_images.append(hog_image_rescaled)

    new_add = add_new_features(images)
    final_features = np.column_stack((train_fd, new_add))
    X = np.array(final_features)

    # X = np.array(train_fd)
    print("feature",X.shape)
    return X,images, hog_images

X,images, h = load_image_files("E:\\CVG\\MicroSuture\\knot_depth_estimation/preprocess_4040", 8)

y0 = np.zeros(1305)
y1 = np.ones(1255)
# 609 images for Class 0, 232 for Class 1.
# y0 = np.zeros(1305)
# y1 = np.ones(667)

# concatenate y0 and y1 to form y
y = []
y = np.concatenate((y1, y0), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print("X train is:",len(X_train))
print("y train is:",len(y_train))
print("X test is:",len(X_test))
print("y test is:",len(y_test))
# define support vector classifier


#grid search for best parameter
param_grid = {'C': [0.01 ,0.1, 1, 10, 100],
              'gamma': [0.01, 0.1, 1, 10, 100],
              'kernel': ['rbf', 'linear', 'poly'],
              'degree': [1, 2, 3, 4, 5]}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)
grid.fit(X_train, y_train)


# print best parameter after tuning
print(grid.best_params_)

# print how our model looks after hyper-parameter tuning
print(grid.best_estimator_)


exit()



#best which we are using
#svm = SVC(kernel='poly',gamma=2,degree=3,C=1, probability=True, random_state=42)

# svm = SVC(kernel='poly',gamma=5,degree=3,C=5, probability=True, random_state=42) #AUC=0.95
#svm = SVC(kernel='poly',gamma=2,degree=3,C=1, probability=True, random_state=42) #addes gamma to earlier params and kernel=poly
#svm = SVC(kernel='poly',gamma=2,degree=3,C=0.0001, probability=True, random_state=42, verbose=False)#f1score(train):0.9969 #auc=0.97

# fit model
svm.fit(X_train, y_train)


y_pred = svm.predict(X_test)
y_tr_pred = svm.predict(X_train)
#print(y_test)

# calculate accuracy
train_accuracy = accuracy_score(y_train, y_tr_pred)
test_accuracy = accuracy_score(y_test, y_pred)
fscore = f1_score(y_test, y_pred)
fscore_train = f1_score(y_train, y_tr_pred)
print('Model accuracy(train): ', train_accuracy)
print('Model accuracy(test): ', test_accuracy)

print("F1 Score(train):", fscore_train)
print("F1 Score(test):", fscore)

#joblib.dump(svm, 'tuned.pkl')
#joblib.dump(svm, 'more_data.pkl')
#joblib.dump(svm, 'energy.pkl')


tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("tp, fp, fn, tn")
print((tp, fp, fn, tn))





probabilities = svm.predict_proba(X_test)

# select the probabilities for label 1.0
y_proba = probabilities[:, 1]

# calculate false positive rate and true positive rate at different thresholds
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_proba, pos_label=1)

# calculate AUC
roc_auc = auc(false_positive_rate, true_positive_rate)

plt.title('Receiver Operating Characteristic')
# plot the false positive rate on the x axis and the true positive rate on the y axis
roc_plot = plt.plot(false_positive_rate,
                    true_positive_rate,
                    label='AUC = {:0.2f}'.format(roc_auc))

plt.legend(loc=0)
plt.plot([0,1], [0,1], ls='--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate');
# plt.show()
