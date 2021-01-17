# TODO : review and document the code
import numpy as np


class PCA:

    def __init__(self, variance_thresh):
        self.variance_thresh = variance_thresh

    def __calculate_pca(self, X):
        X = X.T
        X_centered = X - X.mean()
        cov = np.cov(X_centered)
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        #sotring eigenvalues and their corresponding eigenvectors
        sorted_indices = eigen_values.argsort()[::-1]
        eigen_vectors = eigen_vectors[:, sorted_indices]
        eigen_values = eigen_values[sorted_indices]
        ncomponents = 0
        sum_values = eigen_values.sum() + 1e-11
        cum_sum = 0
        for i in range(1, len(eigen_values)+1):
            cum_sum += eigen_values[i]
            if(cum_sum / sum_values > self.variance_thresh):
                break
            else:
                ncomponents += 1
        #selecting the highest n components
        
        U = eigen_vectors[:, :ncomponents]

        return np.dot(U.T, X_centered).T

    def fit(self, features_list):
        compressed_features_list = []
        for features in features_list:
            compressed_features = self.__calculate_pca(features)
            compressed_features_list.append(compressed_features)
        return compressed_features_list
