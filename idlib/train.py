# TODO : complete train driver implementation
# TODO : get best parameters with grid search
from idlib.dataset.train_loader import TrainLoader
from idlib.feature_extractor.lbp_features import LBPFeatureExtractor
from idlib.feature_extractor.lpq_features import LPQFeatureExtractor
from idlib.feature_extractor.glcm_features import GLCMFeaturesExtractor
from idlib.feature_extractor.pca import PCA
from idlib.preprocessor.form_preparator import FormPreparator

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
    # perform training and grid search on complete data
    # initialize and build dataloader
    print('Initializing dataloader ... \n')
    dataloader = TrainLoader(data_dir=data_dir)
    dataloader.build_dataset()
    # initialize LBP feature extractor
    print('Initializing LBP feature extractor ... \n')
    lbp_extractor = LBPFeatureExtractor()
    # initialize form preparator
    print('Initializing form preparator ... \n')
    form_processor = FormPreparator(denoise=True)
    # get complete train/validation dataset
    print('Reading dataset ... \n')
    xtrain, xvalid, ytrain, yvalid = dataloader.get_complete_data(forms_per_writer=5, test_split=0.2)
    # read and prepare features for train images
    print('Preparing train features ... \n')
    xtrain_features = list()
    ytrain_labels = list()
    for img_name, label in zip(xtrain, ytrain):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        lines, bin_lines = form_processor.prepare_form(img)
        features = lbp_extractor.fit(lines, bin_lines)
        xtrain_features.extend(features)
        ytrain_labels.extend([label]*len(features))
    # read and prepare features for validation images
    print('Preparing test features ... \n')
    xvalid_features = list()
    yvalid_labels = list()
    for img_name, label in zip(xvalid, yvalid):
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        lines, bin_lines = form_processor.prepare_form(img)
        features = lbp_extractor.fit(lines, bin_lines)
        xvalid_features.append(features)
        yvalid_labels.append(label)
    # perform classical model grid search
    print('Performing grid search for best parameters ...')
    svm = SVC()
    parameters = {'kernel': ('linear', 'rbf'), 
                'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'gamma': ('scale', 'auto'),
                'probability': (True, False)}
    clf = GridSearchCV(svm, parameters)
    clf.fit(xtrain_features, ytrain_labels)
    print(f' => best parameters are : {clf.best_params_} \n')
    # get classical model predictions
    print('Performing Inference ... \n')
    predictions = list()
    for form in xvalid_features:
        lines_prob = clf.predict_proba(form)
        lines_prob = np.sum(lines_prob, axis=0)
        predictions.append(clf.classes_[np.argmax(lines_prob)])
    print (f'svm accuracy: {accuracy_score(yvalid_labels, predictions)*100}%')

def sampled_train(data_dir='data/'):
    # perform training and grid search on sampled data
    # intialize train dataloader
    dataloader = TrainLoader(data_dir=data_dir)
    dataloader.build_dataset()
    # initialize LBP feature extractor
    lbp_extractor = LBPFeatureExtractor(radius=3)
    # initialize form preparator
    form_processor = FormPreparator(denoise=True)
    # initialize classifier
    clf = SVC(C=5.0, gamma='auto', probability=True, verbose=True)
    # loop over all test cases
    total_cases = 0
    correct_cases = 0
    for test_case in range(100):
        # read test case images
        xtrain, xvalid, ytrain, yvalid = dataloader.get_data_samples()
        # extract train samples features
        xtrain_features = list()
        ytrain_labels = list()
        for img_name, label in zip(xtrain, ytrain):
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
            lines, bin_lines = form_processor.prepare_form(img)
            features = lbp_extractor.fit(lines, bin_lines)
            xtrain_features.extend(features)
            ytrain_labels.extend([label]*len(features))
        # extract test sample features
        img = cv2.imread(xvalid, cv2.IMREAD_GRAYSCALE)
        lines, bin_lines = form_processor.prepare_form(img)
        test_features = lbp_extractor.fit(lines, bin_lines)
        # train classifier on train samples features
        clf.fit(xtrain_features, ytrain_labels)
        # get predictions
        lines_prob = clf.predict_proba(test_features)
        lines_prob = np.sum(lines_prob, axis=0)
        prediction = np.argmax(lines_prob)
        total_cases += 1
        if yvalid == prediction:
            correct_cases += 1
    print (f'svm accuracy: {float(correct_cases/total_cases)*100}%')
