import numpy as np


class GLCMFeaturesExtractor:
    """
    Greyscale Level Cooccurance Matrix Texture Descriptor
    ...

    An implementation of GLCM texture descriptor.
    ...

    Attributes
    ----------
    img : ndarray
        numpy array of input image
    """
    def __calculate_glcm(self, img):
        # initializing the output array
        glcm_matrix = np.zeros((256, 256))
        # get value of unique pixel values
        unique_pixels = np.unique(img.astype(np.uint8))
        # looping over all combinations
        for first in unique_pixels:
            for second in unique_pixels:
                for i in range(img.shape[0]-1):
                    for j in range(img.shape[1]):
                        # increasing  the value of matched pixels
                        if(img[i][j] == first and img[i+1][j] == second):
                            glcm_matrix[first][second] += 1
        # reshaping the matrix to a single vector
        return glcm_matrix.reshape(-1)

    def fit(self, lines):
        # loop over each lines and extract basic GLCM features
        total_glcm_features = []
        for line in lines:
            glcm_features = self.__calculate_glcm(line)
            total_glcm_features.append(glcm_features)
        return total_glcm_features
