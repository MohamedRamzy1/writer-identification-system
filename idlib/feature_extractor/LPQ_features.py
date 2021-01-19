from __future__ import division
import numpy as np
from scipy.signal import convolve2d

class LPQ:
    """
    Local Phase Quantization Texture Descriptor
    ...

    An implementation of LPQ texture descriptor that uses STFT on a rectangular window_size * window_size neighborhood,
    at different window sizes passed as initial parameter.
    Used to extract LPQ histograms for a list of images.
    ...

    Attributes
    ----------
    window_size : int
    the size of the rectangular neighborhood must be >= 3
        
    """

    def __init__(self, window_size):
        # check if the window size < 3 set it to be 3 ( the minimum accepted value)
        if window_size <3:
            print("window size is set to be 3")
            self.window_size = 3
        else:
            self.window_size = window_size

    def __calculate_LPQ(self,img):
        # convert the image to double
        img=np.float64(img)

        # check a valid window size
        x,y = img.shape
        if y < self.window_size :
            self.window_size = y
        if x < self.window_size:
            self.window_size = x

        # get the radius of the window
        r=(self.window_size-1)/2
        x=np.arange(-r,r+1)[np.newaxis]    

        # construct the kernel parameters to apply STFT to the neighbors 
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*(1/self.window_size)*1j)
        w2=np.conj(w1)   

        ## Run filters to compute the frequency response in the four points.
        filterResp1=convolve2d(convolve2d(img,w0.T,'valid'),w1,'valid')
        filterResp2=convolve2d(convolve2d(img,w1.T,'valid'),w0,'valid')
        filterResp3=convolve2d(convolve2d(img,w1.T,'valid'),w1,'valid')
        filterResp4=convolve2d(convolve2d(img,w1.T,'valid'),w2,'valid')

        # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
        freqResp=np.dstack([filterResp1.real, filterResp1.imag, filterResp2.real, filterResp2.imag, filterResp3.real, filterResp3.imag, filterResp4.real, filterResp4.imag])

        # Perform quantization and compute LPQ codewords
        inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
        LPQdesc=((freqResp>0)*(2**inds)).sum(2)

        # calculate the histogram of the result
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]
        # Normalize histogram
        LPQdesc=LPQdesc/LPQdesc.sum()
        # return the normalized histogram
        return LPQdesc

    def fit(self, lines):
        lpq_features = []
        # loop over the images
        for line in lines:
            # calculate the pca for each one and save the result in lpq_features
            lpq_feature = self.__calculate_LPQ(line)
            lpq_features.append(lpq_feature)

        #return the final data result
        return lpq_features