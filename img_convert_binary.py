import cv2
import numpy as np
# from skimage.filters import (threshold_otsu)
# import glob
# PATH = "images/"
# list_of_dir = glob.glob(PATH + "*")
INPUT_PATH = "D:/Python_Project/CV/data_cin/Y_ml/out.png"


def get_binary_image(image):
    im_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    return im_bw