import cv2 as cv
import numpy as np
from skimage.transform import probabilistic_hough_line
import time
from matplotlib import pyplot as plt
import argparse

def written_extraction(img):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray,50,200)
    lines = probabilistic_hough_line(edges, threshold=10, line_length=120,line_gap=1)
    horizontal_rows = []
    for line in lines:
        if (line[0][1] - line[1][1] <20 and line[0][1] > 420):
            horizontal_rows.append(line[0][1])
    horizontal_rows.sort()
    return img[horizontal_rows[0]+20:horizontal_rows[-1]-20][:]



if __name__ == "__main__":
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-p', '--img_path', type=str, help='path to the image')

    args = argparser.parse_args()
    try:
        img = cv.imread(args.img_path)
    except:
        print("path not exists")
    
    start_time = time.time()
    result = written_extraction(img)
    print("extraction done in  %s seconds" % (time.time() - start_time))
    plt.imshow(result)
    plt.show()
