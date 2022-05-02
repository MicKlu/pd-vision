#!/usr/bin/python3

import sys
import cv2 as cv
import numpy as np
import os

def preprocessing(img):
    img_prep = np.full_like(img, 0)
    cv.intensity_transform.gammaCorrection(img, img_prep, 1)

    img_prep = bluring(img_prep)
    img_prep = sharpening(img_prep)

    # hsv_img = cv.cvtColor(img_prep, cv.COLOR_BGR2HSV)
    # h, s, v = cv.split(hsv_img)

    # v = cv.equalizeHist(v)
    # s = cv.equalizeHist(s)
    # hsv_img = cv.merge((h, s, v))
    # img_prep = cv.cvtColor(img_prep, cv.COLOR_HSV2BGR)

    return img_prep

def bluring(img):
    img_blur = cv.medianBlur(img, 3)
    img_blur = cv.blur(img_blur, (3, 3))
    return img_blur

def sharpening(img):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    # kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    img_sharp = cv.filter2D(img, -1, kernel)
    return img_sharp

def threshold(hsv_img, s_level=23, v_level=96):
    h, s, v = cv.split(hsv_img)

    _, s_thresh = cv.threshold(s, s_level, 255, cv.THRESH_BINARY)
    # _, v_thresh = cv.threshold(v, 109, 255, cv.THRESH_BINARY_INV)
    _, v_thresh = cv.threshold(v, v_level, 255, cv.THRESH_BINARY_INV)

    return s_thresh, v_thresh

def opening_closing(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2))
    img_opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (16, 16))
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
    img_final = cv.morphologyEx(img_opened, cv.MORPH_CLOSE, kernel)

    return img_final

def closing_opening(img):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (16, 16))
    img_closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_final = cv.morphologyEx(img_closed, cv.MORPH_OPEN, kernel)

    return img_final

def full_detect_and_count(img, s_level=23, v_level=96, area_tol=500):
    # Preprocessing
    img_prep = preprocessing(img)

    # Convert BGR -> HSV
    hsv_img = cv.cvtColor(img_prep, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv_img)

    # Threshold S and V
    s_thresh, v_thresh = threshold(hsv_img, s_level=s_level, v_level=v_level)

    # AND S and V
    sv_thresh = cv.bitwise_and(s_thresh, v_thresh)

    # OR S and V
    # sv_thresh = cv.bitwise_or(s_thresh, v_thresh)

    # Erode
    # kernel = np.ones((3, 3), np.uint8)
    # sv_thresh = cv.erode(sv_thresh, kernel)

    # Opening / Closing
    sv_thresh_final = opening_closing(sv_thresh)

    # Closing / Opening
    # sv_thresh_final = closing_opening(sv_thresh)

    # Extract BLOBs
    contours, hierarchy = cv.findContours(sv_thresh_final, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, contours, -1, (0, 0, 255), 3)

    cntLen = 0
    for cnt in contours:
        if cv.contourArea(cnt) > area_tol:
            cv.drawContours(img, [cnt], -1, (255, 0, 0), 3)
            cntLen += 1

    return cntLen, len(contours), (img_prep, s, v, s_thresh, v_thresh, sv_thresh, sv_thresh_final)

def detect_and_count(img, s_level=23, v_level=96, area_tol=500):
    count, _, _ = full_detect_and_count(img, s_level, v_level, area_tol)
    return count

if __name__ == "__main__":
    ### Load image
    imgs_list = os.listdir("imgs")
    
    img_index = 13

    # img = cv.imread("reference_imgs/hsv-test.jpg")
    print(imgs_list[img_index])
    img = cv.imread(f"imgs/{imgs_list[img_index]}")

    ### Resize image
    # img = cv.resize(img, None, fx=0.1, fy=0.1)

    ### Run algorithm
    count, all_count, (img_prep, s, v, s_thresh, v_thresh, sv_thresh, sv_thresh_final) = full_detect_and_count(img, 79, 31)

    ### Put everything to show how image changes
    result_img = np.vstack((
        np.hstack((img, img_prep)),
        np.hstack((cv.cvtColor(s, cv.COLOR_GRAY2BGR), cv.cvtColor(v, cv.COLOR_GRAY2BGR))),
        np.hstack((cv.cvtColor(s_thresh, cv.COLOR_GRAY2BGR), cv.cvtColor(v_thresh, cv.COLOR_GRAY2BGR))),
        np.hstack((cv.cvtColor(sv_thresh, cv.COLOR_GRAY2BGR), cv.cvtColor(sv_thresh_final, cv.COLOR_GRAY2BGR)))
    ))

    ### Display results
    cv.namedWindow("H", cv.WINDOW_NORMAL)
    cv.imshow("H", result_img)

    print(f"All detected objects: {all_count}")
    print(f"Actual detected objects: {count}")

    while True:
        if cv.waitKey() == 27:
            break