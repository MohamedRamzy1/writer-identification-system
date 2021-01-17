from idlib.dataset.train_loader import TrainLoader
from idlib.feature_extractor.lbp_features import LBPFeatureExtractor
from idlib.preprocessor.form_preparator import FormPreparator
from idlib.trainer.model_gird_search import svm_grid_search, knn_grid_search
from idlib.trainer.model_train import svm_train

import cv2
from tqdm import tqdm


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
    xtrain, xvalid, ytrain, yvalid = dataloader.get_complete_data(
        forms_per_writer=5, test_split=0.2
        )
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
    # perform svm grid search
    print('Performing grid search on SVM for best parameters ...')
    svm_grid_search(
        xtrain_features, ytrain_labels, xvalid_features, yvalid_labels
    )
    # perform knn grid search
    print('Performing grid search on KNN for best parameters ...')
    knn_grid_search(
        xtrain_features, ytrain_labels, xvalid_features, yvalid_labels
    )


def sampled_train(data_dir='data/'):
    # perform training and grid search on sampled data
    # intialize train dataloader
    print('Initializing dataloader ... \n')
    dataloader = TrainLoader(data_dir=data_dir)
    dataloader.build_dataset()
    # initialize LBP feature extractor
    print('Initializing LBP feature extractor ... \n')
    lbp_extractor = LBPFeatureExtractor(radius=3)
    # initialize form preparator
    print('Initializing form preparator ... \n')
    form_processor = FormPreparator(denoise=True)
    # loop over all test cases
    print('Performing sampled data training ... \n')
    total_cases = 100
    correct_cases = 0
    for test_case in tqdm(range(total_cases)):
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
        if svm_train(xtrain_features, ytrain_labels, test_features, yvalid):
            correct_cases += 1
    print()
    print(f'model accuracy: {float(correct_cases/total_cases)*100}%')
