#!/usr/bin/python3

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os

def main():
    imgs_list = os.listdir("imgs")

    # for (i, img_file) in enumerate(imgs_list):
    # img = cv.imread(f"imgs/{img_file}")
    img = cv.imread(f"imgs/easy_samples/img1.jpg")
    # h_hist, s_hist, v_hist = get_histogram(img)
    # print("H")
    # print(h_hist[0])
    # print("S")
    # print(s_hist[0])
    # print("V")
    # print(v_hist[0])

    show_histogram(img, fig=0)

def show_histogram(img, color="bgr", fig=None):
    img_rgb, img_hsv = convert_color(img, color)
    h, s, v = cv.split(img_hsv)

    plt.figure(fig)

    plt.subplot(2, 4, 1)
    plt.imshow(img_rgb)
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(h, cmap="gray", vmax=180)
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(s, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(v, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.hist(h.ravel(), 180, [0,180])

    plt.subplot(2, 4, 7)
    plt.hist(s.ravel(), 256, [0,256])

    plt.subplot(2, 4, 8)
    plt.hist(v.ravel(), 256, [0,256])

    plt.show()

def get_histogram(img, color="bgr", normalize=False):
    img_rgb, img_hsv = convert_color(img, color)
    h, s, v = cv.split(img_hsv)

    h_hist = np.histogram(h.ravel(), 180, [0,180])
    s_hist = np.histogram(s.ravel(), 256, [0,256])
    v_hist = np.histogram(v.ravel(), 256, [0,256])

    if normalize:
        h_hist = (h_hist[0] / h_hist[0].max(), h_hist[1])
        s_hist = (s_hist[0] / s_hist[0].max(), s_hist[1])
        v_hist = (v_hist[0] / v_hist[0].max(), v_hist[1])

    return (h_hist, s_hist, v_hist)

def convert_color(img, color="bgr"):
    if color == "bgr":
        img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if color == "hsv":
        img_hsv = img
        img_rgb = cv.cvtColor(img, cv.COLOR_HSV2RGB)
    return img_rgb, img_hsv

if __name__ == "__main__":
    main()