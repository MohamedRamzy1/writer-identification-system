import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from skimage.feature import local_binary_pattern
from sklearn.model_selection import train_test_split

def get_lbp_features(img_path):
    # read image as greyscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # apply gaussian filter (denoising)
    kernel = np.ones((5, 5), np.float32) / 25
    smooth_img = cv2.filter2D(img, -1, kernel)
    # convert to binary
    threshold, bin_img = cv2.threshold(smooth_img, 127.5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # get LBP features
    lbp = local_binary_pattern(smooth_img, 8, 3, 'default')
    # mask LBP features
    lbp[bin_img == 255] = -1
    # get histogram for LBP features
    unique, counts = np.unique(lbp, return_counts=True)
    lbp_counts = dict(zip(unique, counts))
    lbp_hist = np.zeros((256,))
    for i in range(256):
        if float(i) in lbp_counts.keys():
            lbp_hist[i] = lbp_counts[float(i)]
    # normalize LBP histogram
    lbp_hist = np.divide(lbp_hist, np.mean(lbp_hist))
    return lbp_hist

# read forms data
writers_forms = dict()
with open('../data/ascii/forms.txt') as f:
    for i in range(16):
        next(f)
    for line in f:
        entries = line.split()
        if entries[1] in writers_forms.keys():
            writers_forms[entries[1]].append(entries[0])
        else:
            writers_forms[entries[1]] = [entries[0]]

# list image paths and labels
img_paths = list()
img_labels = list()
for id, writer in enumerate(writers_forms.keys()):
    for form in writers_forms[writer]:
        form_path = os.path.join(os.path.join('../data/lines', form[:3]), form)
        form_lines = os.listdir(form_path)
        for line in form_lines:
            img_paths.append(os.path.join(form_path, line))
            img_labels.append(id)

# extract LBP features for each image
img_features = list()
for path in img_paths:
    img_features.append(get_lbp_features(path))

# split data into train and validation
xtrain, xvalid, ytrain, yvalid = train_test_split(img_features, img_labels, stratify=img_labels, random_state=42,
                                                test_size=0.1, shuffle=True)

# SVM classifier
clf = SVC(C=5.0, gamma='auto', probability=True)
clf.fit(xtrain, ytrain)
predictions = clf.predict(xvalid)

print (f'svm accuracy: {accuracy_score(yvalid, predictions)*100}%')

# K-NN classifer
clf = KNeighborsClassifier()
clf.fit(xtrain, ytrain)
predictions = clf.predict(xvalid)

print (f'knn accuracy: {accuracy_score(yvalid, predictions)*100}%')
