import numpy as np
import cv2

class LBPFeatureExtractor:
    """
    Local Binary Pattern Texture Descriptor
    ...

    Vectorized implementation of LBP texture descriptor that uses only 8 points
    at variable radius, which is passed as an initialization parameter.
    Used to extract LBP histograms for a list of images.
    ...

    Attributes
    ----------
    radius : int
        distance of interest points from center point
    """
    def __init__(self, radius=3):
        # initialize parameters
        self.radius = radius

    def _pad_image(self, img):
        # pad grayscale image with radius
        return np.pad(img, self.radius)
    
    def _get_features(self, img):
        # extract LBP features map of a grayscale image
        # get image dimensions
        img_height, img_width = img.shape
        # initialize LBP map with same image dimensions
        lbp_map = np.zeros((img_height, img_width))
        # pad input image with radius
        padded_img = self._pad_image(img)
        # consider all 8 positions (compare whole image instead of only one pixel at a time)
        # TOP
        # mask padded image with original image at top position
        cmp_map = (padded_img[:img_height, self.radius:img_width+self.radius] >= img)
        # multiply comparison map by 2^0 and add to LBP map
        lbp_map[self.radius:, :] += cmp_map[self.radius:, :] * 1
        # TOP-RIGHT
        # mask padded image with original image at top-right position
        cmp_map = (padded_img[:img_height, 2*self.radius:] >= img)
        # multiply comparison map by 2^1 and add to LBP map
        lbp_map[self.radius:, :img_width-self.radius] += cmp_map[self.radius:, :img_width-self.radius] * 2
        # RIGHT
        # mask padded image with original image at right position
        cmp_map = (padded_img[self.radius:img_height+self.radius, 2*self.radius:] >= img)
        # multiply comparison map by 2^2 and add to LBP map
        lbp_map[:, :img_width-self.radius] += cmp_map[:, :img_width-self.radius] * 4
        # BOTTOM-RIGHT
        # mask padded image with original image at bottom-right position
        cmp_map = (padded_img[2*self.radius:, 2*self.radius:] >= img)
        # multiply comparison map by 2^3 and add to LBP map
        lbp_map[:img_height-self.radius, :img_width-self.radius] += cmp_map[:img_height-self.radius, :img_width-self.radius] * 8
        # BOTTOM
        # mask padded image with original image at bottom position
        cmp_map = (padded_img[2*self.radius:, self.radius:img_width+self.radius] >= img)
        # multiply comparison map by 2^4 and add to LBP map
        lbp_map[:img_height-self.radius, :] += cmp_map[:img_height-self.radius, :] * 16
        # BOTTOM-LEFT
        # mask padded image with original image at bottom-left position
        cmp_map = (padded_img[2*self.radius:, :img_width] >= img)
        # multiply comparison map by 2^5 and add to LBP map
        lbp_map[:img_height-self.radius, self.radius:] += cmp_map[:img_height-self.radius, self.radius:] * 32
        # LEFT
        # mask padded image with original image at left position
        cmp_map = (padded_img[self.radius:img_height+self.radius, :img_width] >= img)
        # multiply comparison map by 2^6 and add to LBP map
        lbp_map[:, self.radius:] += cmp_map[:, self.radius:] * 64
        # TOP-LEFT
        # mask padded image with original image at top-left position
        cmp_map = (padded_img[:img_height, :img_width] >= img)
        # multiply comparison map by 2^7 and add to LBP map
        lbp_map[self.radius:, self.radius:] += cmp_map[self.radius:, self.radius:] * 128
        # return extracted LBP map
        return lbp_map

    def _calc_histogram(self, lbp_map, bin_img):
        # calculate masked histogram of LBP map
        # mask LBP map with binary image (only consider black pixels)
        lbp_map[bin_img == 255] = -1
        # get unique elements count of masked LBP map
        unique, counts = np.unique(lbp_map, return_counts=True)
        lbp_counts = dict(zip(unique, counts))
        # extract final 256D feature vector
        lbp_hist = [lbp_counts[float(i)] if float(i) in lbp_counts.keys() else 0 for i in range(256)]
        lbp_hist = np.array(lbp_hist)
        # normalize LBP histogram by its mean
        lbp_hist = np.divide(lbp_hist, np.mean(lbp_hist))
        return lbp_hist

    def fit(self, lines, bin_lines):
        # wrapper function to extract LBP features of a list of lines
        # initialize LBP features list
        lbp_features = list()
        # loop over each line
        for line, bin_line in zip(lines, bin_lines):
            # extract LBP feature map
            lbp_map = self._get_features(line)
            # calculate normalized LBP histogram
            lbp_features.append(self._calc_histogram(lbp_map, bin_line))
        return lbp_features
