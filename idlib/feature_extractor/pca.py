import numpy as np


class PCA:
    """
    Principle Component Analysis Texture Descriptor
    ...

    An implementation of PCA
    that select the number of components based on a certain value,
    which is passed as initial parameter.
    Used to extract PCA of an image.
    ...

    Attributes
    ----------
    ncomponent : int
    the number of taken components from eigen vectors
    """
    def __init__(self, ncomponent):
        self.ncomponent = ncomponent

    def __calculate_pca(self, X):
        X = X.T
        # subtract the mean to center the image around the origin
        X_centered = X - X.mean()
        # calculate the covariance matrix of the zero-mean data.
        cov = np.cov(X_centered)
        # calculate the eigen values and vectors of the covariance matrix
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        # sort the eigen values descendingly
        sorted_indices = eigen_values.argsort()[::-1]
        eigen_vectors = eigen_vectors[:, sorted_indices]
        eigen_values = eigen_values[sorted_indices]
        # check if number of component is valid one ( else take all components)
        if self.ncomponent > len(eigen_values):
            self.ncomponent = len(eigen_values)
        # take real part only from the eigen vectors
        U = eigen_vectors[:, :self.ncomponent].real
        # return the reduced img based on the selected eigen vectors
        return np.dot(U.T, X_centered).T

    def fit(self, features_list):
        compressed_features_list = []
        # loop over the images
        for features in features_list:
            # calculate the pca for each one
            # and save the result in compressed_features_list
            compressed_features = self.__calculate_pca(features)
            compressed_features_list.append(compressed_features)
        # return the final data result
        return compressed_features_list
