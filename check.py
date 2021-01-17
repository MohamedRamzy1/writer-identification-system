import cv2 as cv
import numpy as np
from skimage.transform import probabilistic_hough_line
import time
from matplotlib import pyplot as plt
import argparse
from sklearn import decomposition 
# TODO : review and document the code
#import numpy as np


class PCA:
    """
    Principle Component Analysis Texture Descriptor
    ...

    An implementation of PCA texture descriptor that select the amount of components base on a certain threshold,
    which is passed as initial parameter.
    Used to extract PCA of an image.
    ...

    Attributes
    ----------
    variance_thresh : float
        
    """

    def __init__(self, ncomponent):
        self.ncomponent = ncomponent

    def __calculate_pca(self, X):
        print(X.shape)
        X = X.T
        # subtract the mean to center the image around the origin
        X = X - X.mean()
        print(X)
        # calculate the covariance matrix of the zero-mean data.
        cov = np.cov(X)
        #print(cov)
        # calculate the eigen values and vectors of the covariance metrix 
        eigen_values, eigen_vectors = np.linalg.eig(cov)
        #sort the eigen values descendingly 
        sorted_indices = eigen_values.argsort()[::-1]
        eigen_vectors = eigen_vectors[:, sorted_indices]
        eigen_values = eigen_values[sorted_indices]
        # get the number of taken components based on the passed threshold
        #initialize it with zero
        if self.ncomponent > len(eigen_values):
            self.ncomponent = len(eigen_values)
        # take the calculated number of components
        U = eigen_vectors[:, :self.ncomponent].real

        # return the reduced img based on the selected eigen vectors
        return np.dot(U.T, X).T

    def fit(self, features_list):
        compressed_features_list = []
        # loop over the images
        for features in features_list:
            # calculate the pca for each one and save the result in compressed_features_list
            compressed_features = self.__calculate_pca(features)
            compressed_features_list.append(compressed_features)
            
        #return the final data result
        return compressed_features_list


if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-p', '--img_path', type=str, help='path to the image')

    args = argparser.parse_args()
    # read the image
    pca_ex = PCA(3400)
    try:
        img = cv.imread(args.img_path)
    except:
        print("path not exists")
    #start timer to get the execution time
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    print(gray.shape)
    result = pca_ex.fit([gray])
    start_time = time.time()

    plt.imshow(result[0].astype(np.uint8))
    plt.show()

        

