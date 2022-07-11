#!/usr/bin/python3

import os
from alg.custom import CustomHsvBlobAlgorithm
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import sys
import math

def colorize(small, big):
    h = (big/255*120 + small/255*30).astype("uint8")
    sv = np.where(h == 0, 0, 255).astype("uint8")
    return cv.cvtColor(cv.merge((h, sv, sv)), cv.COLOR_HSV2RGB)

def plot_morphs(alg: CustomHsvBlobAlgorithm, s_blobs: tuple, v_blobs: tuple, s_morphs: tuple, v_morphs: tuple, results: tuple):
    sdims = (4, 4)

    max_sv = np.ones_like(alg.img_s_thresh) * 255

    plt.figure()
    
    # S
    plt.subplot(*sdims, 1)
    plt.imshow(alg.img_s_thresh, cmap="gray")
    plt.axis("off")
    
    # V
    plt.subplot(*sdims, 3)
    plt.imshow(alg.img_v_thresh, cmap="gray")
    plt.axis("off")

    # S-segments
    plt.subplot(*sdims, 2)
    plt.imshow(colorize(s_blobs[1], s_blobs[0]))
    plt.axis("off")
    
    # V-segments
    plt.subplot(*sdims, 4)
    plt.imshow(colorize(v_blobs[1], v_blobs[0]))
    plt.axis("off")

    # S-small
    plt.subplot(*sdims, 5)
    plt.imshow(s_blobs[1], cmap="gray")
    plt.axis("off")

    # S-big
    plt.subplot(*sdims, 6)
    plt.imshow(s_blobs[0], cmap="gray")
    plt.axis("off")

    # V-small
    plt.subplot(*sdims, 7)
    plt.imshow(v_blobs[1], cmap="gray")
    plt.axis("off")

    # V-big
    plt.subplot(*sdims, 8)
    plt.imshow(v_blobs[0], cmap="gray")
    plt.axis("off")

    # S-small morphed
    plt.subplot(*sdims, 9)
    plt.imshow(s_morphs[1], cmap="gray")
    plt.axis("off")

    # S-big morphed
    plt.subplot(*sdims, 10)
    plt.imshow(s_morphs[0], cmap="gray")
    plt.axis("off")

    # V-small morphed
    plt.subplot(*sdims, 11)
    plt.imshow(v_morphs[1], cmap="gray")
    plt.axis("off")

    # V-big morphed
    plt.subplot(*sdims, 12)
    plt.imshow(v_morphs[0], cmap="gray")
    plt.axis("off")

    # S morphed
    img_s_morphed = results[0]
    plt.subplot(*sdims, 13)
    plt.imshow(img_s_morphed, cmap="gray")
    plt.axis("off")

    # V morphed
    img_v_morphed = results[1]
    plt.subplot(*sdims, 15)
    plt.imshow(img_v_morphed, cmap="gray")
    plt.axis("off")

    # SV morphed AND
    img_sv_morphed = results[2]
    plt.subplot(*sdims, 16)
    plt.imshow(img_sv_morphed, cmap="gray")
    plt.axis("off")

    # SV morphed OR
    img_sv_morphed = results[3]
    plt.subplot(*sdims, 14)
    plt.imshow(img_sv_morphed, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

def separate(img):
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    areas = np.array([ cv.contourArea(c) for c in contours ])
    avg = np.average(areas)
    
    print(areas.min())
    print(areas.max())
    print(avg)

    big_blobs = np.zeros_like(img)
    for c in contours:
        if cv.contourArea(c) > avg:
            cv.drawContours(big_blobs, [c], 0, 255, -1)

    small_blobs = img - big_blobs

    return (big_blobs, small_blobs)

def fill_holes(img_hollow):
    _, mask = cv.threshold(img_hollow, 0, 255, cv.THRESH_BINARY)
    cont, hier = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    img_filled = mask
    for c in cont:
        cv.drawContours(img_filled, [c], 0, 255, -1)

    return img_filled

def closing_with_filling(img, ksize=(3, 3)):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
    img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)
    img = fill_holes(img)
    return cv.morphologyEx(img, cv.MORPH_ERODE, kernel)

def remove_small_blobs(img, thresh=1):
    out = np.copy(img)
    contours, _ = cv.findContours(out, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if cv.contourArea(c) <= thresh:
            cv.drawContours(out, [c], 0, 0, -1)

    return out

if __name__ == "__main__":
    
    ### Load image
    img_difficulty = "easy" # easy, moderate, hard, extreme
    img_index = 4

    img_path = f"imgs/{img_difficulty}_samples/img{img_index}.jpg"
    
    print()

    alg = CustomHsvBlobAlgorithm(img_path)
    count = alg.count()

    s_blobs = separate(alg.img_s_thresh)
    v_blobs = separate(alg.img_v_thresh)

    # BEGIN S morph
    
    img_s_morphed_big = np.copy(s_blobs[0])
    img_s_morphed_small = np.copy(s_blobs[1])

    # BEGIN S-small morph

    # img_s_morphed_small = remove_small_blobs(img_s_morphed_small)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    img_s_morphed_small = cv.morphologyEx(img_s_morphed_small, cv.MORPH_DILATE, kernel, iterations=3)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    img_s_morphed_small = cv.morphologyEx(img_s_morphed_small, cv.MORPH_ERODE, kernel, iterations=1)

    # END S-small morph

    # BEGIN S-big morph

    img_s_morphed_big = closing_with_filling(img_s_morphed_big, (17, 17))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    img_s_morphed_big = cv.morphologyEx(img_s_morphed_big, cv.MORPH_OPEN, kernel, iterations=1)

    # END S-big morph

    # split = 8

    # h = math.ceil(np.size(img_s_morphed_small, 0) / split)
    # w = math.ceil(np.size(img_s_morphed_small, 1) / split)

    # print()
    # print(w, h)

    # plt.figure()
    # plt.ion()
    # plt.show()

    # for i in range(0, split):
    #     for j in range(0, split):
    #         print(f"{j * h}:{(j + 1) * h}, {i * w}:{(i + 1) * w}")
    #         subimg = img_s_morphed_small[j * h:(j + 1) * h, i * w:(i + 1) * w]
    #         cv.imshow("dsa", subimg)
    #         pixel_count = np.sum(np.where(subimg == 255, 1, 0))
    #         print(pixel_count)

    #         # if pixel_count >= 1000:
    #         #     kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    #         #     subimg = cv.morphologyEx(subimg, cv.MORPH_ERODE, kernel)
    #         #     img_s_morphed_small[j * h:(j + 1) * h, i * w:(i + 1) * w] = subimg

    #         # cv.imshow("ewq", subimg)

    #         xy = np.where(subimg == 255)

    #         plt.clf()
    #         plt.subplot(2, 1, 1)
    #         plt.hist(xy[1], bins=w, range=[0, w])

    #         plt.subplot(2, 1, 2)
    #         plt.hist(xy[0], bins=h, range=[0, h])

    #         pass

    

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # img_s_morphed_small = cv.morphologyEx(img_s_morphed_small, cv.MORPH_DILATE, kernel, iterations=5)
    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # img_s_morphed_small = cv.morphologyEx(img_s_morphed_small, cv.MORPH_ERODE, kernel)
    # img_s_morphed_small = cv.morphologyEx(img_s_morphed_small, cv.MORPH_CLOSE, kernel, iterations=10)
    # img_s_morphed_small = cv.morphologyEx(img_s_morphed_small, cv.MORPH_DILATE, kernel, iterations=10)
    # img_s_morphed_small = cv.morphologyEx(img_s_morphed_small, cv.MORPH_ERODE, kernel)

    # img_s_morphed_big = dilate_with_filling(img_s_morphed_big)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    # img_s_morphed_big = cv.morphologyEx(img_s_morphed_big, cv.MORPH_ERODE, kernel, iterations=1)

    # END S morph

    # BEGIN V morph
    
    img_v_morphed_big = np.copy(v_blobs[0])
    img_v_morphed_small = np.copy(v_blobs[1])

    # BEGIN V-small morph

    img_v_morphed_small = remove_small_blobs(img_v_morphed_small)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    img_v_morphed_small = cv.morphologyEx(img_v_morphed_small, cv.MORPH_DILATE, kernel, iterations=3)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    img_v_morphed_small = cv.morphologyEx(img_v_morphed_small, cv.MORPH_ERODE, kernel, iterations=1)

    # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    # img_v_morphed_small = cv.morphologyEx(img_v_morphed_small, cv.MORPH_DILATE, kernel, iterations=5)

    # END V-small morph

    # BEGIN V-big morph

    img_v_morphed_big = closing_with_filling(img_v_morphed_big, (17, 17))
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
    img_v_morphed_big = cv.morphologyEx(img_v_morphed_big, cv.MORPH_OPEN, kernel, iterations=1)

    # END V-big morph

    # END V morph

    s_morphs = (img_s_morphed_big, img_s_morphed_small)
    v_morphs = (img_v_morphed_big, img_v_morphed_small)


    # S morphed
    img_s_morphed = cv.bitwise_or(*s_morphs)

    # V morphed
    img_v_morphed = cv.bitwise_or(*v_morphs)

    # SV morphed AND
    img_sv_morphed_and = cv.bitwise_and(img_s_morphed, img_v_morphed)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    img_sv_morphed_and = cv.morphologyEx(img_sv_morphed_and, cv.MORPH_CLOSE, kernel)

    # SV morphed OR
    img_sv_morphed_or = cv.bitwise_or(img_s_morphed, img_v_morphed)

    results = (img_s_morphed, img_v_morphed, img_sv_morphed_and, img_sv_morphed_or)

    plot_morphs(alg, s_blobs, v_blobs, s_morphs, v_morphs, results)