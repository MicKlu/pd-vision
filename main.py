#!/usr/bin/python3

import alg as a
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import hist

def plot_results(alg: a.BaseHsvBlobAlgorithm, backend="matplotlib"):

    img_output = np.copy(alg.img_original_bgr)
    cv.drawContours(img_output, alg.blobs, -1, (0, 0, 255), 3)
    cv.drawContours(img_output, alg.valid_blobs, -1, (255, 0, 0), 3)

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
            np.hstack((cv.cvtColor(alg.img_morphed, cv.COLOR_GRAY2BGR), empty_img, empty_img))
        ))

        cv.namedWindow("Results", cv.WINDOW_NORMAL)
        cv.imshow("Results", result_img)

        while True:
            if cv.waitKey() == 27:
                break

if __name__ == "__main__":

    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    imgs_list = os.listdir(f"imgs/{img_difficulty}_samples")
    
    img_index = 0
    img_path = f"imgs/{img_difficulty}_samples/{imgs_list[img_index]}"
    print(img_path)

    print()

    ### Reference algorithm
    s_threshs = [38, 86, 39, 159, 40, 65, 64, 30, 77, 78, 39, 52, 123, 79]
    v_threshs = [104, 156, 100, 72, 93, 64, 37, 37, 36, 36, 99, 94, 102, 31]

    ref_alg = a.ReferenceAlgorithm(img_path, s_threshs[img_index], v_threshs[img_index])
    count = ref_alg.count()

    print("### Reference algorithm ###")
    print(f"All blobs: {len(ref_alg.blobs)}")
    print(f"Valid blobs / Counted objects: {count}\n")

    # plot_results(ref_alg, backend="opencv")

    ### Median-based thresholding algorithm
    alg = a.MedianBasedThresholdingAlgorithm(img_path)
    count = alg.count()

    print("### Median-based thresholding algorithm ###")
    print(f"All blobs: {len(alg.blobs)}")
    print(f"Valid blobs / Counted objects: {count}\n")

    # plot_results(alg, backend="opencv")

    ### Median-based filtering algorithm
    alg = a.MedianBasedFilteringAlgorithm(img_path)
    count = alg.count()

    print("### Median-based filtering algorithm ###")
    print(f"All blobs: {len(alg.blobs)}")
    print(f"Valid blobs / Counted objects: {count}\n")

    # plot_results(alg, backend="opencv")

    ### Median-based thresholding OR algorithm
    alg = a.MedianBasedThresholdingORAlgorithm(img_path)
    count = alg.count()

    print("### Median-based thresholding OR algorithm ###")
    print(f"All blobs: {len(alg.blobs)}")
    print(f"Valid blobs / Counted objects: {count}\n")

    # plot_results(alg, backend="opencv")

    ### Median-based filtering OR algorithm
    alg = a.MedianBasedFilteringORAlgorithm(img_path)
    count = alg.count()

    print("### Median-based filtering OR algorithm ###")
    print(f"All blobs: {len(alg.blobs)}")
    print(f"Valid blobs / Counted objects: {count}\n")

    # plot_results(alg, backend="opencv")

    ### Increment-based filtering OR algorithm
    alg = a.IncrementBasedFilteringAlgorithm(img_path)
    count = alg.count()

    print("### Increment-based filtering OR algorithm ###")
    print(f"All blobs: {len(alg.blobs)}")
    print(f"Valid blobs / Counted objects: {count}\n")

    plot_results(alg, backend="opencv")

    h = alg.img_prep_h
    s = cv.bitwise_and(alg.img_prep_s, alg.img_s_thresh)
    v = cv.bitwise_and(alg.img_prep_v, alg.img_v_thresh)

    hist.show_histogram(cv.merge((h,s,v)), color="hsv")
