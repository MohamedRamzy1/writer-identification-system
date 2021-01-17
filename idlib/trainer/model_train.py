from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import numpy as np


def svm_train(xtrain, ytrain, xvalid, yvalid):
    # train svm classifier on train samples features
    # define svm classfier
    clf = SVC(C=5.0, gamma='auto', probability=True, verbose=True)
    # fit model to train data
    clf.fit(xtrain, ytrain)
    # get predictions
    lines_prob = clf.predict_proba(xvalid)
    lines_prob = np.sum(lines_prob, axis=0)
    prediction = np.argmax(lines_prob)
    # check whether the test sample if correct or not
    if yvalid == prediction:
        return True
    else:
        return False


def knn_train(xtrain, ytrain, xvalid, yvalid):
    # train knn classifier on train samples features
    # define knn classfier
    clf = KNeighborsClassifier(n_jobs=-1)
    # fit model to train data
    clf.fit(xtrain, ytrain)
    # get predictions
    lines_prob = clf.predict_proba(xvalid)
    lines_prob = np.sum(lines_prob, axis=0)
    prediction = np.argmax(lines_prob)
    # check whether the test sample if correct or not
    if yvalid == prediction:
        return True
    else:
        return False


def rf_train(xtrain, ytrain, xvalid, yvalid):
    # train rf classifier on train samples features
    # define rf classfier
    clf = RandomForestClassifier(n_jobs=-1, verbose=True)
    # fit model to train data
    clf.fit(xtrain, ytrain)
    # get predictions
    lines_prob = clf.predict_proba(xvalid)
    lines_prob = np.sum(lines_prob, axis=0)
    prediction = np.argmax(lines_prob)
    # check whether the test sample if correct or not
    if yvalid == prediction:
        return True
    else:
        return False


def lr_train(xtrain, ytrain, xvalid, yvalid):
    # train lr classifier on train samples features
    # define lr classfier
    clf = LogisticRegression(
            C=1.0, max_iter=100, n_jobs=-1, verbose=True
        )
    # fit model to train data
    clf.fit(xtrain, ytrain)
    # get predictions
    lines_prob = clf.predict_proba(xvalid)
    lines_prob = np.sum(lines_prob, axis=0)
    prediction = np.argmax(lines_prob)
    # check whether the test sample if correct or not
    if yvalid == prediction:
        return True
    else:
        return False


def nb_train(xtrain, ytrain, xvalid, yvalid):
    # train nb classifier on train samples features
    # define nb classfier
    clf = MultinomialNB(alpha=1.0)
    # fit model to train data
    clf.fit(xtrain, ytrain)
    # get predictions
    lines_prob = clf.predict_proba(xvalid)
    lines_prob = np.sum(lines_prob, axis=0)
    prediction = np.argmax(lines_prob)
    # check whether the test sample if correct or not
    if yvalid == prediction:
        return True
    else:
        return False
