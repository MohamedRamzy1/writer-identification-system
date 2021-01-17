import cv2 as cv
import numpy as np
from skimage.transform import probabilistic_hough_line
import time
from matplotlib import pyplot as plt
import argparse

def written_extraction(img):
    """
    written_extraction
    ...

    extract the hand written part of the image
    ...

    Attributes
    ----------
    img : numpy array
        the rgb image to work with   
    """
    #first convert the img to grayscale
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #get the edges of the img using canny edge detection
    edges = cv.Canny(gray,50,200)
    # extract all lines in the image
    lines = cv.HoughLinesP(edges,1,np.pi/180 ,threshold=8, minLineLength=80,maxLineGap=1)
    # hold the horizontal lines
    horizontal_rows = []
    # loop over the found lines to get the horizontal ones
    for line in lines:
        # take the horizntal line starting from the second one 
        if (line[0][1] - line[0][3] <20 and line[0][1] > 420):
            horizontal_rows.append(line[0][1])
    # sort the lines to take the first and last one
    horizontal_rows.sort()
    # crop the image based on the horizontal lines with margin 20 pixels up and down
    return img[horizontal_rows[0]+20:horizontal_rows[-1]-20][:]



if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-p', '--img_path', type=str, help='path to the image')

    args = argparser.parse_args()
    # read the image
    try:
        img = cv.imread(args.img_path)
    except:
        print("path not exists")
    #start timer to get the execution time
    start_time = time.time()
    # extract the hand written parts
    result = written_extraction(img)
    # print the time algorithm takes
    print("extraction done in  %s seconds" % (time.time() - start_time))
    # plot the result
    plt.imshow(result)
    plt.show()
