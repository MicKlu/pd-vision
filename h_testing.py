#!/usr/bin/python3

import os
from alg.custom import CustomHsvBlobAlgorithm
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import sys

def plot_h_mask(alg: CustomHsvBlobAlgorithm):
    original_bgr = alg.img_original_bgr
    h_uneq = alg.img_prep_h_uneq
    h_uneq_thresh = cv.bitwise_and(h_uneq, alg.h_mask)

    h, s, v = cv.split(cv.cvtColor(original_bgr, cv.COLOR_BGR2HSV))

    plt.figure("h_mask")

    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(original_bgr, cv.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(2, 3, 4)

    h_uneq_out = cv.cvtColor(cv.merge([
        np.where(h_uneq_thresh == 0, 120, 30).astype("uint8"),
        np.full_like(s, 255),
        np.clip((v * 1.9), 0, 255).astype("uint8")
    ]), cv.COLOR_HSV2RGB)

    plt.imshow(h_uneq_out)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(h_uneq, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.hist(h_uneq.ravel(), 180, [0,180])

    plt.subplot(2, 3, 5)
    plt.imshow(h_uneq_thresh, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 3, 6)
    plt.hist(h_uneq_thresh.ravel(), 180, [0,180])

    # cv.imwrite("h_testing_out/h_uneq_out.png", cv.cvtColor(h_uneq_out, cv.COLOR_RGB2BGR))

    plt.show()

def plot_masked_sv(alg: CustomHsvBlobAlgorithm):
    plt.figure("masked_sv")

    subplot_dimens = (3, 4)

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 1)
    plt.imshow(alg.img_prep_s_uneq, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 2)
    plt.hist(alg.img_prep_s_uneq.ravel(), 256, [0,256])

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 3)
    plt.imshow(alg.img_prep_v_uneq, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 4)
    plt.hist(alg.img_prep_v_uneq.ravel(), 256, [0,256])

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 5)
    plt.imshow(alg.img_prep_s, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 6)
    plt.hist(alg.img_prep_s.ravel(), 256, [0,256])

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 7)
    plt.imshow(alg.img_prep_v, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 8)
    plt.hist(alg.img_prep_v.ravel(), 256, [0,256])

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 9)
    plt.imshow(alg.img_s_h_masked, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 10)
    plt.hist(alg.img_s_h_masked.ravel(), 256, [1,256])

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 11)
    plt.imshow(alg.img_v_h_masked, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 12)
    plt.hist(alg.img_v_h_masked.ravel(), 256, [1,256])

    plt.show()

def plot_thresholded_sv(alg: CustomHsvBlobAlgorithm):
    plt.figure("masked_sv")

    subplot_dimens = (3, 2)

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 1)
    plt.imshow(alg.img_s_h_masked, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 2)
    plt.imshow(alg.img_v_h_masked, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 3)
    plt.imshow(alg.img_s_thresh, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 4)
    plt.imshow(alg.img_v_thresh, cmap="gray")
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
    # plt.imshow(cv.morphologyEx(alg.img_v_thresh, cv.MORPH_TOPHAT, kernel), cmap="gray")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    
    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    img_index = 2

    img_path = f"imgs/{img_difficulty}_samples/img{img_index}.jpg"
    
    print()

    alg = CustomHsvBlobAlgorithm(img_path)
    count = alg.count()

    # h_uneq is optimal
    # plot_h_mask(alg)

    plot_masked_sv(alg)
    plot_thresholded_sv(alg)
