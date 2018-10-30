import numpy as np
import cv2


def get_contours_by_color(image, color):
    color = np.array(color, dtype="uint8")
    mask = cv2.inRange(image, color, color)
    output = cv2.bitwise_and(image, image, mask=mask)

    imgray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(imgray, 1, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def is_contour_bad(contour):
    area = cv2.contourArea(contour)
    if area < 20:
        return True
    else:
        return False
