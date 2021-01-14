import numpy as np
import cv2

class FormPreparator:
    """
    Form Preparator class
    ...

    prepare the form image by :
    1) denoising the form image.
    2) clipping out the printed parts.
    3) dividing image into separate lines
    ...

    Attributes
    ----------
    denoise : bool
        whether to perform denoising or not
    """
    def __init__(self, denoise=True):
        # intialize parameters
        self.denoise = denoise

    def binarize_image(self, img):
        # apply OTSU threshold on a grayscale image
        _, bin_img = cv2.threshold(img, 127.5, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return bin_img

    def perform_denoise(self, img):
        # apply gaussian filter for denoising
        kernel = np.ones((5, 5), np.float32) / 25
        smooth_img = cv2.filter2D(img, -1, kernel)
        return smooth_img

    def clip_form(self, img):
        # TODO : perform better form clipping
        # get the written part from the form
        img = img[int(0.25*img.shape[0]):int(0.7*img.shape[0]), int(0.1*img.shape[1]):int(0.9*img.shape[1])]
        return img

    def segment_lines(self, gray_img, bin_img, min_line_height=10):
        # split a form image into lines
        # define padding function
        def pad_with(vector, pad_width, iaxis, kwargs):
            # pad with zeros
            pad_value = kwargs.get('padder', 0)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value
        # count
        ones = np.sum(bin_img,1)
        # histogram
        mean = np.mean(ones * 1.0) / 5
        histo = (ones > mean) * 1
        # get rising and falling edges from histogram
        shifted = np.roll(histo, -1, 0)
        shifted[-1] = histo[-1]
        edges = histo - shifted
        print(type((edges == -1).nonzero()))
        rising_indices = np.array((edges == -1).nonzero()).flatten()
        falling_indices = np.array((edges == 1).nonzero()).flatten()
        if len(falling_indices) < 2 or len(rising_indices) < 2:
            return np.array([gray_img]), np.array([bin_img])
        # make starting with rising not falling 
        if falling_indices[0] < rising_indices[0]:
            falling_indices = falling_indices[1:]
        # make ending with falling not rising 
        if rising_indices[-1] > falling_indices[-1]:
            rising_indices = rising_indices[:-1]
        # cut image on histo
        gray_lines = []
        bin_lines = []
        line_count = min(rising_indices.shape[0], falling_indices.shape[0])
        for i in range(line_count):
            line_height = falling_indices[i] - rising_indices[i]
            # split gray with padding
            start_split = max(rising_indices[i] - line_height//3, 0)
            end_split = min(falling_indices[i] + line_height//3, gray_img.shape[0])
            gray_line = gray_img[start_split:end_split]
            # split binary
            bin_line = bin_img[rising_indices[i]:falling_indices[i]]
            # pad binary
            bin_line = np.pad(bin_line, line_height//3, pad_with)
            # filter if less than 10 pixels
            if line_height > min_line_height:
                gray_lines.append(gray_line)
                bin_lines.append(1-bin_line)
        gray_lines = np.array(gray_lines)
        bin_lines = np.array(bin_lines)
        return gray_lines, bin_lines

    def prepare_form(self, img):
        # prepare a form image
        # perform denoising (if applicable)
        if self.denoise:
            img = self.perform_denoise(img)
        # clip out written parts
        clipped_img = self.clip_form(img)
        # binarize image
        bin_img = self.binarize_image(clipped_img)
        # segment lines from form image
        lines, bin_lines = self.segment_lines(clipped_img, bin_img)
        return lines, bin_lines
