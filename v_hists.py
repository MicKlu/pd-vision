#!/usr/bin/python3

from alg.custom import CustomHsvBlobAlgorithm
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from hist import get_single_channel_histogram
from scipy import signal

if __name__ == "__main__":
    
    plt.figure()

    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    for img_index in range(0, 5):
        img_path = f"imgs/{img_difficulty}_samples/img{img_index}.jpg"
    
        print()
        print(f"==== {img_path} ====")
        print()

        alg = CustomHsvBlobAlgorithm(img_path)
        count = alg.count()

        sp_dim = (3, 4)

        plt.subplot(sp_dim[0], sp_dim[1], 2 * img_index + 1)
        plt.imshow(alg.img_v_h_masked, cmap="gray")
        plt.axis("off")

        plt.subplot(sp_dim[0], sp_dim[1], 2 * img_index + 2)
        plt.hist(alg.img_v_h_masked.ravel(), 256, [1,256])

        v_hist = get_single_channel_histogram(alg.img_v_h_masked)[1:]

        peaks, _ = signal.find_peaks(np.append(v_hist, 0), np.average(v_hist))
        plt.plot(peaks+1, v_hist[peaks], "x")
        m = np.max(peaks)
        plt.plot(m+1, v_hist[m], "rx")

    plt.show()