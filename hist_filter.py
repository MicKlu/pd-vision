#!/usr/bin/python3

import cv2 as cv
import numpy as np
import hist

def median_thresh(hist):
    thresh_val = 0.404786979067976 * np.median(hist[0]) + 0.0118168544770851
    error = np.abs((thresh_val - hist[0]))
    thresh_val_q = error.min()
    thresh = error.tolist().index(thresh_val_q)
    return thresh, thresh_val


if __name__ == "__main__":
    img = cv.imread("imgs/img13.jpg")
    h_hist, s_hist, v_hist = hist.get_histogram(img, normalize=True)
    s_thresh, s_thresh_val = median_thresh(s_hist)
    v_thresh, v_thresh_val = median_thresh(v_hist)

    print(s_thresh, s_thresh_val)
    print(v_thresh, v_thresh_val)

    hist.show_histogram(img)