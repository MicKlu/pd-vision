#!/usr/bin/python3

import os
from alg.custom import CustomHsvBlobAlgorithm
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import sys

def plot_h_mask(img, step, i):
    
    (h,s,v) = cv.split(img)

    # plt.figure("h_mask")

    # plt.subplot(1, 2, 1)

    # _, thresh = cv.threshold(step["img"], 1, 255, cv.THRESH_BINARY)
    # h_masked = cv.bitwise_and(h, thresh)

    # out = cv.cvtColor(cv.merge([
    #     np.where(h_masked == 0, 120, 30).astype("uint8"),
    #     np.full_like(s, 255),
    #     np.clip((v * 1.9), 0, 255).astype("uint8")
    # ]), cv.COLOR_HSV2RGB)

    # cv.imwrite(f"out/thresh_h_{i}.png", cv.cvtColor(out, cv.COLOR_RGB2BGR))

    # plt.imshow(out)
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # hist = get_single_channel_histogram(step["img"], 180)[1:]
    # plt.bar(np.arange(1, 180), hist, 1)
    # if "pix" in step:
    #     plt.bar([step["pix"]], [np.max(hist)], 1, color="r")

    # plt.show()


    plt.figure(figsize=(4,2), dpi=72)
    hist = get_single_channel_histogram(step["img"], 180)[1:]
    plt.bar(np.arange(1, 180), hist, 1)
    if "pix" in step:
        plt.bar([step["pix"]], [np.max(hist)], 1, color="r")

    plt.xlabel(r"Poziom jasności $r_k$")
    plt.ylabel(r"Liczba pikseli $h(r_k)$")
    plt.xlim(1, 179)
    plt.tight_layout(pad=0.01)

    plt.savefig(f"out/thresh_h_{i}_hist.png")
    

def get_single_channel_histogram(channel, bins=256):
    hist = np.histogram(channel.ravel(), bins, [0,bins])
    hist = hist[0]
    return hist

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
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 5)
    plt.imshow(alg.img_s_morphed, cmap="gray")
    plt.axis("off")

    plt.subplot(subplot_dimens[0], subplot_dimens[1], 6)
    plt.imshow(alg.img_v_morphed, cmap="gray")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    
    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    img_index = 1

    img_path = f"imgs/{img_difficulty}_samples/img{img_index}.jpg"
    
    print()

    alg = CustomHsvBlobAlgorithm(img_path)
    count = alg.count()

    for (i, step) in enumerate(alg.h_steps):
        plot_h_mask(alg.img_original_bgr, step, i)
        input()

    # h_uneq is optimal
    # plot_h_mask(alg)

    # plot_masked_sv(alg)
    # plot_thresholded_sv(alg)
