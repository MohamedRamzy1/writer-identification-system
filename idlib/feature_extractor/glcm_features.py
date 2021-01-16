# TODO : review and document the code
import numpy as np

class GLCMFeaturesExtractor:

    def __calculate_glcm(self, img):
        glcm_matrix = np.zeros((256,256))
        unique_pixels = np.unique(img.astype(np.uint8))
        for first in unique_pixels:
            for second in unique_pixels:
                for i in range(img.shape[0]-1):
                    for j in range(img.shape[1]):
                        if(img[i][j] == first and img[i+1][j] == second):
                            glcm_matrix[first][second] += 1
        return glcm_matrix.reshape(-1)

    def fit(self, lines):
        total_glcm_features = []
        for line in lines:
            glcm_features = self.__calculate_glcm(line)
            total_glcm_features.append(glcm_features)
        return total_glcm_features
                

                
            
                
                