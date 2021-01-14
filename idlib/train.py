# TODO : complete train driver implementation
# TODO : get best parameters with grid search
from idlib.dataset.train_loader import TrainLoader
from idlib.feature_extractor.lbp_features import LBPFeatureExtractor
from idlib.feature_extractor.lpq_features import LPQFeatureExtractor
from idlib.feature_extractor.glcm_features import GLCMFeaturesExtractor
from idlib.feature_extractor.pca import PCA
from idlib.preprocessor.form_preparation import FormPreparator

import torch
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import cv2
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def complete_train(data_dir='data/'):
    # initialize and build dataloader
    dataloader = TrainLoader(data_dir=data_dir)
    dataloader.build_dataset()
    # initialize LBP feature extractor
    lbp_extractor = LBPFeatureExtractor()
    # get complete train/validation dataset
    xtrain, xvalid, ytrain, yvalid = dataloader.get_complete_data(forms_per_writer=5, test_split=0.2)
    # read and prepare features for train images
    xtrain_features = list()
    ytrain_labels = list()
    for img_name, label in zip(xtrain, ytrain):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        lines, bin_lines = FormPreparator.prepare_form(img)
        features = lbp_extractor.fit(lines, bin_lines)
        xtrain_features.extend(features)
        ytrain_labels.extend([label]*len(features))
    # read and prepare features for validation images
    xvalid_features = list()
    yvalid_labels = list()
    for img_name, label in zip(xvalid, yvalid):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        lines, bin_lines = FormPreparator.prepare_form(img)
        features = lbp_extractor.fit(lines, bin_lines)
        xvalid_features.append(features)
        yvalid_labels.append(label)
    # perform classical model grid search
    svm = SVC()
    parameters = {'kernel': ('linear', 'rbf'), 
                'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'gamma': ('scale', 'auto'),
                'probability': (True, False)}
    clf = GridSearchCV(svm, parameters)
    clf.fit(xtrain_features, ytrain_labels)
    # get classical model predictions
    predictions = list()
    for form in xvalid_features:
        lines_prob = clf.predict_proba(form)
        lines_prob = np.sum(lines_prob, axis=0)
        predictions.append(clf.classes_[np.argmax(lines_prob)])
    print (f'svm accuracy: {accuracy_score(yvalid_labels, predictions)*100}%')

def sampled_train(data_dir='data/'):
    # initialize and build dataloader
    dataloader = TrainLoader(data_dir=data_dir)
    dataloader.build_dataset()
    # initialize LBP feature extractor
    lbp_extractor = LBPFeatureExtractor()
    for case in range(100):
        # get complete train/validation dataset
        xtrain, xvalid, ytrain, yvalid = dataloader.get_data_samples(num_writers=3, num_train_samples=2)
        # read and prepare features for train images
        xtrain_features = list()
        ytrain_labels = list()
        for img_name, label in zip(xtrain, ytrain):
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            lines, bin_lines = FormPreparator.prepare_form(img)
            features = lbp_extractor.fit(lines, bin_lines)
            xtrain_features.extend(features)
            ytrain_labels.extend([label]*len(features))
        # read and prepare features for validation images
        xvalid_features = list()
        yvalid_labels = list()
        for img_name, label in zip(xvalid, yvalid):
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            lines, bin_lines = FormPreparator.prepare_form(img)
            features = lbp_extractor.fit(lines, bin_lines)
            xvalid_features.append(features)
            yvalid_labels.append(label)
        # perform classical model grid search
        clf = SVC(C=5.0, gamma='auto', probability=True, verbose=True)
        clf.fit(xtrain_features, ytrain_labels)
        # get classical model predictions
        predictions = list()
        for form in xvalid_features:
            lines_prob = clf.predict_proba(form)
            lines_prob = np.sum(lines_prob, axis=0)
            predictions.append(clf.classes_[np.argmax(lines_prob)])
    print (f'svm accuracy: {accuracy_score(yvalid_labels, predictions)*100}%')
