import numpy as np
import cv2

class LBPFeatureExtractor:
    def __init__(self, radius):
        self.radius = radius

    def _binarize_line(self, gs_img):
        _, bin_img = cv2.threshold(gs_img, 127.5, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bin_img

    def _pad_image(self, img):
        return np.pad(img, self.radius)
    
    def _get_features(self, img):
        img_height, img_width = img.shape
        lbp_map = np.zeros((img_height, img_width))
        padded_img = self._pad_image(img)
        # top
        cmp_map = (padded_img[:img_height, self.radius:img_width+self.radius] >= img)
        lbp_map[self.radius:, :] += cmp_map[self.radius:, :] * 1
        # top-right
        cmp_map = (padded_img[:img_height, 2*self.radius:] >= img)
        lbp_map[self.radius:, :img_width-self.radius] += cmp_map[self.radius:, :img_width-self.radius] * 2
        # right
        cmp_map = (padded_img[self.radius:img_height+self.radius, 2*self.radius:] >= img)
        lbp_map[:, :img_width-self.radius] += cmp_map[:, :img_width-self.radius] * 4
        # bottom-right
        cmp_map = (padded_img[2*self.radius:, 2*self.radius:] >= img)
        lbp_map[:img_height-self.radius, :img_width-self.radius] += cmp_map[:img_height-self.radius, :img_width-self.radius] * 8
        # bottom
        cmp_map = (padded_img[2*self.radius:, self.radius:img_width+self.radius] >= img)
        lbp_map[:img_height-self.radius, :] += cmp_map[:img_height-self.radius, :] * 16
        # bottom-left
        cmp_map = (padded_img[2*self.radius:, :img_width] >= img)
        lbp_map[:img_height-self.radius, self.radius:] += cmp_map[:img_height-self.radius, self.radius:] * 32
        # left
        cmp_map = (padded_img[self.radius:img_height+self.radius, :img_width] >= img)
        lbp_map[:, self.radius:] += cmp_map[:, self.radius:] * 64
        # top-left
        cmp_map = (padded_img[:img_height, :img_width] >= img)
        lbp_map[self.radius:, self.radius:] += cmp_map[self.radius:, self.radius:] * 128
        return lbp_map

    def _calc_histogram(self, lbp_map, bin_img):
        lbp_map[bin_img == 255] = -1
        unique, counts = np.unique(lbp_map, return_counts=True)
        lbp_counts = dict(zip(unique, counts))
        lbp_hist = [lbp_counts[float(i)] for i in range(256) if float(i) in lbp_counts.keys()]
        lbp_hist = np.array(lbp_hist)
        lbp_hist = np.divide(lbp_hist, np.mean(lbp_hist))
        return lbp_hist

    def fit(self, lines):
        lbp_features = list()
        for line in lines:
            bin_line = self._binarize_line(line)
            lbp_map = self._get_features(line)
            lbp_features.append(self._calc_histogram(lbp_map, bin_line))
        return lbp_features
