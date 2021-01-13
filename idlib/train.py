from dataset.train_loader import TrainLoader
from feature_extractor.lbp_features import LBPFeatureExtractor
from feature_extractor.lpq_features import LPQFeatureExtractor
from feature_extractor.glcm_features import GLCMFeaturesExtractor
from feature_extractor.pca import PCA
from preprocessor.form_preparation import FormPreparator

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
    pass

def sampled_train(data_dir='data/'):
    pass
