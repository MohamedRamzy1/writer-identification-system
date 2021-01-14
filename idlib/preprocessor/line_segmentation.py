import torch
import numpy as np

class LineSegmentor:
    """
    Segment out lines from form image and extract them.
    """
    @staticmethod
    def split_in_lines(img, bin_img, min_line_height=10):
        # split a form image into lines
        # TODO : improve line segmentation
        def pad_with(vector, pad_width, iaxis, kwargs):
            # pad with zeros
            pad_value = kwargs.get('padder', 0)
            vector[:pad_width[0]] = pad_value
            vector[-pad_width[1]:] = pad_value
        # binarize image
        img = 1 - ((img > 128) * 1)
        # convert to torch tensor
        img_tensor = torch.tensor(img)
        # count ones along vertical axis
        ones = torch.sum(img_tensor,1)
        # calculate histogram
        mean = torch.mean(ones * 1.0) / 5
        histo = (ones > mean) * 1
        # get rising and falling edges from histogram
        shifted = torch.roll(histo, -1, 0)
        shifted[-1] = histo[-1]
        edges = histo - shifted
        rising_indices = torch.flatten((edges == -1).nonzero())
        falling_indices = torch.flatten((edges == 1).nonzero())
        if len(falling_indices) < 2 or len(rising_indices) < 2:
            return img
        # make starting with rising not falling 
        if falling_indices[0] < rising_indices[0]:
            falling_indices = falling_indices[1:]
        # make ending with falling not rising 
        if rising_indices[-1] > falling_indices[-1]:
            rising_indices = rising_indices[:-1]
        # cut image on histo
        lines = []
        line_count = min(rising_indices.size()[0], falling_indices.size()[0])
        for i in range(line_count):
            line = img[rising_indices[i]:falling_indices[i]]
            line_height = line.shape[0]
            line = np.pad(line, line.shape[0]//3, pad_with)
            # filter if less than 10 pixels
            if line_height > min_line_height:
                lines.append(1-line)
        return lines
