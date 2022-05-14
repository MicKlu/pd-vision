#!/usr/bin/python3

import os
import alg as a
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

def plot_results(original_bgr, h_uneq, h_eq, h_uneq_thresh, h_eq_thresh):

    h, s, v = cv.split(cv.cvtColor(original_bgr, cv.COLOR_BGR2HSV))

    plt.figure()

    plt.subplot(3, 4, 1)
    plt.imshow(cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB))

    plt.subplot(3, 4, 2)

    h_uneq_out = cv.cvtColor(cv.merge([
        np.where(h_uneq_thresh == 0, 120, 30).astype("uint8"),
        np.full_like(s, 255),
        np.clip((v * 1.9), 0, 255).astype("uint8")
    ]), cv.COLOR_HSV2RGB)

    plt.imshow(h_uneq_out)

    plt.subplot(3, 4, 3)

    h_eq_out = cv.cvtColor(cv.merge([
        np.where(h_eq_thresh == 0, 120, 30).astype("uint8"),
        np.full_like(s, 255),
        np.clip((v * 1.9), 0, 255).astype("uint8")
    ]), cv.COLOR_HSV2RGB)

    plt.imshow(h_eq_out)

    plt.subplot(3, 4, 4)

    _, m1 = cv.threshold(h_uneq_thresh, 0, 255, cv.THRESH_BINARY)
    _, m2 = cv.threshold(h_eq_thresh, 0, 255, cv.THRESH_BINARY)

    h_mixed_thresh = cv.bitwise_or(m1, m2)

    h_mixed_out = cv.cvtColor(cv.merge([
        np.where(h_mixed_thresh == 0, 120, 30).astype("uint8"),
        np.full_like(s, 255),
        np.clip((v * 1.9), 0, 255).astype("uint8")
    ]), cv.COLOR_HSV2RGB)

    plt.imshow(h_mixed_out)

    plt.subplot(3, 4, 5)
    plt.imshow(h_uneq, cmap="gray")

    plt.subplot(3, 4, 6)
    plt.hist(h_uneq.ravel(), 180, [0,180])

    plt.subplot(3, 4, 7)
    plt.imshow(h_uneq_thresh, cmap="gray")

    plt.subplot(3, 4, 8)
    plt.hist(h_uneq_thresh.ravel(), 180, [0,180])

    plt.subplot(3, 4, 9)
    plt.imshow(h_eq, cmap="gray")

    plt.subplot(3, 4, 10)
    plt.hist(h_eq.ravel(), 180, [0,180])

    plt.subplot(3, 4, 11)
    plt.imshow(h_eq_thresh, cmap="gray")

    plt.subplot(3, 4, 12)
    plt.hist(h_eq_thresh.ravel(), 180, [0,180])

    cv.imwrite("h_testing_out/h_uneq_out.png", cv.cvtColor(h_uneq_out, cv.COLOR_RGB2BGR))
    cv.imwrite("h_testing_out/h_eq_out.png", cv.cvtColor(h_eq_out, cv.COLOR_RGB2BGR))
    cv.imwrite("h_testing_out/h_mixed_out.png", cv.cvtColor(h_mixed_out, cv.COLOR_RGB2BGR))

    plt.show()

def get_histogram(img_channel):
    hist = np.histogram(img_channel.ravel(), 180, [0,180])
    hist = hist[0] / hist[0].max()
    return hist

def reduce_colors(img_h, close=False):

    img_hist = get_histogram(img_h)[1:]

    pix_val = img_hist.tolist().index(img_hist.max()) + 1
    mask = np.where(img_h == pix_val, 0, 255).astype("uint8")
    print(f"pix_val: {pix_val}")
    print(np.sum(mask == 0))

    img_thresh = cv.bitwise_and(img_h, mask)

    if close:
        _, mask = cv.threshold(img_thresh, 0, 255, cv.THRESH_BINARY)
        mask = closing_opening(mask, close_ksize=(3, 3), open_ksize=(3, 3))
        img_thresh = cv.bitwise_and(img_h, mask)

    return img_thresh

def closing_opening(img, close_ksize: tuple = (16, 16), open_ksize: tuple = (5, 5), close_iterations=1, open_iterations=1):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, close_ksize)
    img_closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=close_iterations)

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, open_ksize)
    img_opened = cv.morphologyEx(img_closed, cv.MORPH_OPEN, kernel, iterations=open_iterations)

    return img_opened

def fill_holes(img_original, img_hollow):
    _, mask = cv.threshold(img_hollow, 0, 255, cv.THRESH_BINARY)
    cont, hier = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    for c in cont:
        cv.drawContours(mask, [c], 0, 255, -1)

    return cv.bitwise_and(img_original, mask)

if __name__ == "__main__":
    
    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    imgs_list = os.listdir(f"imgs/{img_difficulty}_samples")
    
    img_index = 0
    img_path = f"imgs/{img_difficulty}_samples/{imgs_list[img_index]}"
    
    print()

    alg = a.CustomHsvBlobAlgorithm(img_path)
    count = alg.count()

    h_uneq = alg.img_prep_h_uneq
    h_eq = alg.img_prep_h

    # h_uneq = cv.medianBlur(h_uneq, 3)
    # h_eq = cv.medianBlur(h_eq, 3)

    # Move all zeros to 179 (almost same red color)
    _, m = cv.threshold(h_uneq, 0, 179, cv.THRESH_BINARY_INV)
    h_uneq = cv.bitwise_or(h_uneq, m)

    _, m = cv.threshold(h_eq, 0, 179, cv.THRESH_BINARY_INV)
    h_eq = cv.bitwise_or(h_eq, m)

    # Remove most significant color
    h_uneq_thresh = reduce_colors(h_uneq)
    h_eq_thresh = reduce_colors(h_eq)

    for i in range(0, 3):   # max 5; optimal 3
        h_uneq_thresh = reduce_colors(h_uneq_thresh)
        h_eq_thresh = reduce_colors(h_eq_thresh)

    h_uneq_thresh = reduce_colors(h_uneq_thresh, close=True)
    h_eq_thresh = reduce_colors(h_eq_thresh, close=True)

    h_uneq_thresh = fill_holes(h_uneq, h_uneq_thresh)
    h_eq_thresh = fill_holes(h_eq, h_eq_thresh)

    # h_uneq is optimal
    plot_results(alg.img_original_bgr, h_uneq, h_eq, h_uneq_thresh, h_eq_thresh)
