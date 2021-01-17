# TODO : review and document the code
import scipy.fft as fft
import numpy as np


class LPQFeatureExtractor:

    def __init__(self, window_size=5, nbins=256):
        self.window_size = window_size
        self.nbins = nbins

    def __calculate_lpq(self, img):
        lpq_codes = []
        for start_x in range(img.shape[0]):
            for start_y in range(img.shape[1]):
                end_x = start_x + self.window_size
                end_y = start_y + self.window_size
                if(end_x >= img.shape[0] or end_y >= img.shape[1]):
                    break
                sub_img = img[start_x:end_x, start_y: end_y]
                sub_h = sub_img.shape[0]
                sub_w = sub_img.shape[1]
                cx = sub_h // 2
                cy = sub_w // 2
                fft2_img = fft.fft2(sub_img)

                rsp1 = fft2_img[cx, cy+1]
                rsp2 = fft2_img[cx+1, cy]
                rsp3 = fft2_img[cx+1, cy+1]
                rsp4 = fft2_img[cx-1, cy+1]

                aggregated_rsp = np.array([
                    rsp1.real, rsp2.real, rsp3.real, rsp4.real,
                    rsp1.imag, rsp2.imag, rsp3.imag, rsp4.imag
                    ])
                bit_vector = list(aggregated_rsp > 0)

                lpq_code = 0

                i = len(bit_vector) - 1

                while(i >= 0):
                    lpq_code += bit_vector[i] * (2**i)
                    i -= 1

                lpq_codes.append(lpq_code)

        histogram, _ = np.histogram(np.array(lpq_codes), bins=self.nbins)

        return histogram

    def fit(self, lines):
        lqp_features = []
        for line in lines:
            histogram = self.__calculate_lpq(line.squeeze())
            lqp_features.append(histogram)
        return lqp_features
