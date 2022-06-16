#!/usr/bin/python3

import alg as a
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import hist

def plot_results(alg: a.BaseHsvBlobAlgorithm, backend="matplotlib"):

    img_output = np.copy(alg.img_original_bgr)
    cv.drawContours(img_output, alg.blobs, -1, (0, 0, 255), 3)
    
    for l in alg.valid_blobs:
        pt1 = (l["stats"][cv.CC_STAT_LEFT], l["stats"][cv.CC_STAT_TOP])
        pt2 = (pt1[0] + l["stats"][cv.CC_STAT_WIDTH], pt1[1] + l["stats"][cv.CC_STAT_HEIGHT])
        cv.rectangle(img_output, pt1, pt2, (255, 0, 0), 3)
    
    if backend == "matplotlib":
        plt.figure()

        rows, cols = (4, 3)

        plt.subplot(rows, cols, 1)
        plt.imshow(cv.cvtColor(alg.img_original_bgr, cv.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Input")

        plt.subplot(rows, cols, 2)
        plt.imshow(cv.cvtColor(alg.img_prep_bgr, cv.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Preprocessed")

        plt.subplot(rows, cols, 3)
        plt.imshow(cv.cvtColor(img_output, cv.COLOR_BGR2RGB))
        plt.axis("off")
        plt.title("Output")

        plt.subplot(rows, cols, 4)
        plt.imshow(alg.img_prep_h, cmap="gray")
        plt.axis("off")
        plt.title("H plane")

        plt.subplot(rows, cols, 5)
        plt.imshow(alg.img_prep_s, cmap="gray")
        plt.axis("off")
        plt.title("S plane")

        plt.subplot(rows, cols, 6)
        plt.imshow(alg.img_prep_v, cmap="gray")
        plt.axis("off")
        plt.title("V plane")

        plt.subplot(rows, cols, 7)
        plt.imshow(alg.img_h_thresh, cmap="gray")
        plt.axis("off")
        plt.title("H plane thresholded")

        plt.subplot(rows, cols, 8)
        plt.imshow(alg.img_s_thresh, cmap="gray")
        plt.axis("off")
        plt.title("S plane thresholded")

        plt.subplot(rows, cols, 9)
        plt.imshow(alg.img_v_thresh, cmap="gray")
        plt.axis("off")
        plt.title("V plane thresholded")

        plt.subplot(rows, cols, 10)
        plt.imshow(alg.img_bitwise, cmap="gray")
        plt.axis("off")
        plt.title("Bitwise operations")

        plt.subplot(rows, cols, 11)
        plt.imshow(alg.img_morphed, cmap="gray")
        plt.axis("off")
        plt.title("Morphology operations")

        plt.subplots_adjust(hspace=0.2, wspace=0.1)
        # plt.tight_layout()
        plt.show()
    elif backend == "opencv":
        empty_img = np.full_like(alg.img_original_bgr, 0)

        result_img = np.vstack((
            np.hstack((alg.img_original_bgr, alg.img_prep_bgr, img_output)),
            np.hstack((cv.cvtColor(alg.img_prep_h, cv.COLOR_GRAY2BGR), cv.cvtColor(alg.img_prep_s, cv.COLOR_GRAY2BGR), cv.cvtColor(alg.img_prep_v, cv.COLOR_GRAY2BGR))),
            np.hstack((cv.cvtColor(alg.img_h_thresh, cv.COLOR_GRAY2BGR), cv.cvtColor(alg.img_s_thresh, cv.COLOR_GRAY2BGR), cv.cvtColor(alg.img_v_thresh, cv.COLOR_GRAY2BGR))),
            np.hstack((cv.cvtColor(alg.img_bitwise, cv.COLOR_GRAY2BGR), cv.cvtColor(alg.img_morphed, cv.COLOR_GRAY2BGR), empty_img))
        ))

        cv.namedWindow("Results", cv.WINDOW_NORMAL)
        cv.imshow("Results", result_img)
        cv.imwrite("/tmp/results.png", result_img)

        while True:
            if cv.waitKey() == 27:
                break

if __name__ == "__main__":

    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    
    img_index = 1
    img_path = f"imgs/{img_difficulty}_samples/img{img_index}.jpg"
    print(img_path)

    print()

    ### Reference algorithm
    s_threshs = { "easy": [17, 125, 39, 159, 40], "moderate": [65, 64, 30, 77, 78, 39, 52, 123, 79]}
    v_threshs = { "easy": [94, 156, 100, 72, 93], "moderate": [64, 37, 37, 36, 36, 99, 94, 102, 31]}

    ref_alg = a.ReferenceAlgorithm(img_path, s_threshs[img_difficulty][img_index], v_threshs[img_difficulty][img_index])
    count = ref_alg.count()

    print("### Reference algorithm ###")
    print(f"All blobs: {len(ref_alg.blobs)}")
    print(f"Valid blobs / Counted objects: {count}\n")

    plot_results(ref_alg, backend="opencv")
    hist.show_histogram(ref_alg.img_prep_hsv, color="hsv")