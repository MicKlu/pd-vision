#!/usr/bin/python3

import os
from alg.custom import CustomHsvBlobAlgorithm
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import sys

def plot_results(original_bgr, h_uneq, h_uneq_thresh):
    h, s, v = cv.split(cv.cvtColor(original_bgr, cv.COLOR_BGR2HSV))

    plt.figure()

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

if __name__ == "__main__":
    
    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    img_index = 1

    img_path = f"imgs/{img_difficulty}_samples/img{img_index}.jpg"
    
    print()

    alg = CustomHsvBlobAlgorithm(img_path)
    count = alg.count()

    # h_uneq is optimal
    plot_results(alg.img_original_bgr, alg.img_prep_h_uneq, alg.h_mask)
