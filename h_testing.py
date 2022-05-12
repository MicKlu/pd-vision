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

    plt.imshow(cv.cvtColor(cv.merge([
        np.where(h_uneq_thresh == 0, 120, 30).astype("uint8"),
        np.full_like(s, 255),
        np.clip((v * 1.9), 0, 255).astype("uint8")
    ]), cv.COLOR_HSV2RGB))

    plt.subplot(3, 4, 3)
    plt.imshow(cv.cvtColor(cv.merge([
        np.where(h_eq_thresh == 0, 120, 30).astype("uint8"),
        np.full_like(s, 255),
        np.clip((v * 1.9), 0, 255).astype("uint8")
    ]), cv.COLOR_HSV2RGB))

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

    plt.show()

def get_histogram(img_channel):
    hist = np.histogram(img_channel.ravel(), 180, [0,180])
    hist = hist[0] / hist[0].max()
    return hist

def reduce_colors(h_uneq, h_eq):
    h_uneq_hist = get_histogram(h_uneq)[1:]
    h_eq_hist = get_histogram(h_eq)[1:]
    
    pix_val = h_uneq_hist.tolist().index(h_uneq_hist.max()) + 1
    h_uneq_mask = np.where(h_uneq == pix_val, 0, 255).astype("uint8")
    print(f"h_uneq pix_val: {pix_val}")

    pix_val = h_eq_hist.tolist().index(h_eq_hist.max()) + 1
    print(f"h_eq pix_val: {pix_val}")
    h_eq_mask = np.where(h_eq == pix_val, 0, 255).astype("uint8")
    print(np.sum(h_eq_mask == 0))
    
    h_uneq_thresh = cv.bitwise_and(h_uneq, h_uneq_mask)
    h_eq_thresh = cv.bitwise_and(h_eq, h_eq_mask)
    return h_uneq_thresh, h_eq_thresh

if __name__ == "__main__":
    
    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    imgs_list = os.listdir(f"imgs/{img_difficulty}_samples")
    
    img_index = 3
    img_path = f"imgs/{img_difficulty}_samples/{imgs_list[img_index]}"
    
    print()

    alg = a.CustomHsvBlobAlgorithm(img_path)
    count = alg.count()

    h_uneq = alg.img_prep_h_uneq
    h_eq = alg.img_prep_h

    # Move all zeros to 179 (almost same red color)
    _, m = cv.threshold(h_uneq, 0, 179, cv.THRESH_BINARY_INV)
    h_uneq = cv.bitwise_or(h_uneq, m)

    _, m = cv.threshold(h_eq, 0, 179, cv.THRESH_BINARY_INV)
    h_eq = cv.bitwise_or(h_eq, m)

    # Remove most significant color
    h_uneq_thresh, h_eq_thresh = reduce_colors(h_uneq, h_eq)

    for i in range(0, 6):   # max 6 best for now
        h_uneq_thresh, h_eq_thresh = reduce_colors(h_uneq_thresh, h_eq_thresh)

    plot_results(alg.img_original_bgr, h_uneq, h_eq, h_uneq_thresh, h_eq_thresh)
