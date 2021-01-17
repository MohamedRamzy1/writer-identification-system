import torch
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


class SupportVectorMachine:
    """
    Support Vector Machine Classifier
    """
    def __init__(self):
        # initialize classifier
        self.clf = SVC(C=5.0, gamma='auto', probability=True, verbose=False)

    def fit(self, x, y):
        # train classifier
        self.clf.fit(x, y)

    def predict(self, x):
        # predict class (label) of data point x
        return self.clf.predict(x)

    def predict_proba(self, x):
        # predict class (probability) of data point x
        return self.clf.predict_proba(x)


class KNearestNeighbors:
    """
    KNN Classifier
    """
    def __init__(self):
        # initialize classifier
        self.clf = KNeighborsClassifier(n_jobs=-1)

    def fit(self, x, y):
        # train classifier
        self.clf.fit(x, y)

    def predict(self, x):
        # predict class (label) of data point x
        return self.clf.predict(x)

    def predict_proba(self, x):
        # predict class (probability) of data point x
        return self.clf.predict_proba(x)


class RandomForest:
    """
    Random Forest Classifier
    """
    def __init__(self):
        # initialize classifier
        self.clf = RandomForestClassifier(n_jobs=-1, verbose=False)

    def fit(self, x, y):
        # train classifier
        self.clf.fit(x, y)

    def predict(self, x):
        # predict class (label) of data point x
        return self.clf.predict(x)

    def predict_proba(self, x):
        # predict class (probability) of data point x
        return self.clf.predict_proba(x)


class GradientBoosting:
    """
    Linear Classifier With Gradient Boosting
    """
    def __init__(self):
        # initialize classifier
        self.clf = xgb.XGBClassifier(
            max_depth=7, n_estimators=200,
            colsample_bytree=0.8, subsample=0.8,
            n_jobs=-1, learning_rate=0.1
        )

    def fit(self, x, y):
        # train classifier
        self.clf.fit(x, y)

    def predict(self, x):
        # predict class (label) of data point x
        return self.clf.predict(x)

    def predict_proba(self, x):
        # predict class (probability) of data point x
        return self.clf.predict_proba(x)


class LogisticRegressor:
    """
    Logistic Regression Classifier
    """
    def __init__(self):
        # initialize classifier
        self.clf = LogisticRegression(
            C=1.0, max_iter=100, n_jobs=-1, verbose=False
        )

    def fit(self, x, y):
        # train classifier
        self.clf.fit(x, y)

    def predict(self, x):
        # predict class (label) of data point x
        return self.clf.predict(x)

    def predict_proba(self, x):
        # predict class (probability) of data point x
        return self.clf.predict_proba(x)


class NaiveBayes:
    """
    Naive Bayes Classifier
    """
    def __init__(self):
        # initialize classifier
        self.clf = MultinomialNB(alpha=1.0)

    def fit(self, x, y):
        # train classifier
        self.clf.fit(x, y)

    def predict(self, x):
        # predict class (label) of data point x
        return self.clf.predict(x)

    def predict_proba(self, x):
        # predict class (probability) of data point x
        return self.clf.predict_proba(x)


class Perceptron(torch.nn.Module):
    """
    Two-layer Perceptron Classifier
    """
    def __init__(self, input_dim=256, n_classes=3):
        # initialize classifier layers and activation functions
        super(Perceptron, self).__init__()
        # fully-connected layers
        self.linear_first = torch.nn.Linear(input_dim, int(input_dim/2))
        self.linear_second = torch.nn.Linear(int(input_dim/2), n_classes)
        # activations
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # forward pass
        outputs = self.relu(self.linear_first(x))
        outputs = self.linear_second(outputs)
        outputs = self.sigmoid(outputs)
        return outputs
