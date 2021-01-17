from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def svm_grid_search(xtrain, ytrain, xvalid, yvalid):
    # perform support vector machine grid search
    # define support vector classifier
    svm = SVC()
    # define parameters list
    parameters = {
        'kernel': ('linear', 'rbf'),
        'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'gamma': ('scale', 'auto'),
        'probability': (True, False)
    }
    # define grid search classifier
    clf = GridSearchCV(svm, parameters)
    # fit model to train data
    clf.fit(xtrain, ytrain)
    print(f' => best parameters are : {clf.best_params_} \n')
    # get model predictions
    print('Performing Inference ... \n')
    predictions = list()
    for form in xvalid:
        # perform majority vote
        lines_prob = clf.predict_proba(form)
        lines_prob = np.sum(lines_prob, axis=0)
        predictions.append(clf.classes_[np.argmax(lines_prob)])
    print(f'svm accuracy: {accuracy_score(yvalid, predictions)*100}% \n')


def knn_grid_search(xtrain, ytrain, xvalid, yvalid):
    # perform k nearset neighbors grid search
    # define knn classifier
    knn = KNeighborsClassifier()
    # define parameters list
    parameters = {'n_neighbors': [3, 5, 7]}
    # define grid search classifier
    clf = GridSearchCV(knn, parameters)
    # fit model to train data
    clf.fit(xtrain, ytrain)
    print(f' => best parameters are : {clf.best_params_} \n')
    # get model predictions
    print('Performing Inference ... \n')
    predictions = list()
    for form in xvalid:
        # perform majority vote
        lines_prob = clf.predict_proba(form)
        lines_prob = np.sum(lines_prob, axis=0)
        predictions.append(clf.classes_[np.argmax(lines_prob)])
    print(f'knn accuracy: {accuracy_score(yvalid, predictions)*100}% \n')
