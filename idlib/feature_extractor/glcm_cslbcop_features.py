import numpy as np
from skimage.feature import greycomatrix


class GLCMCSLBCoPFeaturesExtractor:
    '''
    GLCM_CSLBCoP Feature based on CSLBP
    ----
    based on this paper: https://www.researchgate.net/publication/281559563_Center_Symmetric_Local_Binary_Co-occurrence_Pattern_for_Texture_Face_and_Bio-medical_Image_Retrieval
    ----

    Attributes
    ----------
    LBP_maps : list of matrices
        LBP maps of lines to apply GLCM_CSLBCoP on
    '''

    def _combine_GLCMs(self, GLCM1, GLCM2, GLCM3, GLCM4):
        GLCM1_flattened = GLCM1.flatten()
        GLCM2_flattened = GLCM2.flatten()
        GLCM3_flattened = GLCM3.flatten()
        GLCM4_flattened = GLCM4.flatten()
        FV = np.concatenate([GLCM1_flattened, GLCM2_flattened, GLCM3_flattened, GLCM4_flattened])
        return FV

    def _Calculate_GLCM_CSLBCoP(self, LBP_map):
        # GLCM matrix on distances (0, 1) and angles (0, -45, -90, -135)
        GLCM = greycomatrix(LBP_map, [1, 2], [0, -np.pi/4, -np.pi/2, -3*np.pi/4], levels=256)

        # extract different combinations of GLCM
        GLCM_1_0 = GLCM[:, :, 0, 0]
        GLCM_1_45 = GLCM[:, :, 0, 1]
        GLCM_1_90 = GLCM[:, :, 0, 2]
        GLCM_1_135 = GLCM[:, :, 0, 3]

        GLCM_2_0 = GLCM[:, :, 1, 0]
        GLCM_2_45 = GLCM[:, :, 1, 1]
        GLCM_2_90 = GLCM[:, :, 1, 2]
        GLCM_2_135 = GLCM[:, :, 1, 3]

        # Feature Vectors
        # FV1: combining GLCMs of distance 1 and all angles
        FV1 = self._combine_GLCMs(GLCM_1_0, GLCM_1_45, GLCM_1_90, GLCM_1_135)
        # FV2: combining GLCMs of distance 2 and all angles
        FV2 = self._combine_GLCMs(GLCM_2_0, GLCM_2_45, GLCM_2_90, GLCM_2_135)
        # FV3: combining GLCMs of distance (1, 2) and of angles (0, -45)
        FV3 = self._combine_GLCMs(GLCM_1_0, GLCM_1_45, GLCM_2_0, GLCM_2_45)
        # FV3: combining GLCMs of distance (1, 2) and of angles (0, -90)
        FV4 = self._combine_GLCMs(GLCM_1_0, GLCM_1_90, GLCM_2_0, GLCM_2_90)
        # FV: combining the FVs
        FV = np.concatenate([FV1, FV2, FV3, FV4])
        
        return FV

    def fit(self, LBP_maps):
        total_GLCM_CSLBCoP_features = []
        for LBP_map in LBP_maps:
            GLCM_CSLBCoP_features = self._Calculate_GLCM_CSLBCoP(LBP_map)
            total_GLCM_CSLBCoP_features.append(GLCM_CSLBCoP_features)
        return total_GLCM_CSLBCoP_features
