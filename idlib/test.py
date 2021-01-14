from idlib.dataset.test_loader import TestLoader
from idlib.classifier import models
from idlib.feature_extractor.lbp_features import LBPFeatureExtractor
from idlib.preprocessor.form_preparator import FormPreparator

import numpy as np
import os
import time

def test(data_dir='data/test_samples/'):
    # intialize test dataloader
    dataloader = TestLoader(data_dir)
    test_cases = dataloader.list_test_cases()
    # initialize LBP feature extractor
    lbp_extractor = LBPFeatureExtractor(radius=3)
    # initialize form preparator
    form_processor = FormPreparator(denoise=True)
    # initialize classifier
    clf = models.SupportVectorMachine()
    # initialize output files
    pred_file = open(os.path.join(data_dir, 'results.txt'), 'w+')
    time_file = open(os.path.join(data_dir, 'time.txt'), 'w+')
    # loop over all test cases
    for test_case in test_cases:
        # read test case images
        xtrain, ytrain, xtest = dataloader.get_test_case(test_case)
        # start timer
        start_time = time.time()
        # extract train samples features
        xtrain_features = list()
        ytrain_labels = list()
        for img, label in zip(xtrain, ytrain):
            lines, bin_lines = form_processor.prepare_form(img)
            features = lbp_extractor.fit(lines, bin_lines)
            xtrain_features.extend(features)
            ytrain_labels.extend([label]*len(features))
        # extract test sample features
        lines, bin_lines = form_processor.prepare_form(xtest)
        test_features = lbp_extractor.fit(lines, bin_lines)
        # train classifier on train samples features
        clf.fit(xtrain_features, ytrain_labels)
        # get predictions
        lines_prob = clf.predict_proba(test_features)
        lines_prob = np.sum(lines_prob, axis=0)
        prediction = np.argmax(lines_prob)
        # calculate elapsed time
        total_time = time.time() - start_time
        # write outputs
        pred_file.write(str(prediction+1)+'\n')
        time_file.write(str(total_time)+'\n')
    # close output files
    pred_file.close()
    time_file.close()
