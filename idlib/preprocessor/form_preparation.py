import numpy as np
from line_segmentation import LineSegmentor

class FormPreparator:
    @staticmethod
    def prepare_form(img):
        # clip
        # TODO : perform better form clipping
        img = img[int(0.25*img.shape[0]):int(0.7*img.shape[0]), int(0.1*img.shape[1]):int(0.9*img.shape[1])]
        # segment lines
        return LineSegmentor.split_in_lines(img)
