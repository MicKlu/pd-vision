#!/usr/bin/python3

import sys

sys.path.append("..")

import cv2 as cv
from ref_alg.alg import detect_and_count
import os

imgs_list = os.listdir("imgs")


with open("_calc_tresh_results.csv", "w") as f:
    f.write("thresh_value")
    for (i, img_file) in enumerate(imgs_list):
        f.write(f";{img_file}_s_count")
        f.write(f";{img_file}_v_count")
    f.write("\n")

    for thresh in range(0, 256):
        counts = []
        for (i, img_file) in enumerate(imgs_list):
            img = cv.imread(f"imgs/{img_file}")
            count = detect_and_count(img, s_level=thresh, v_level=255)
            counts.append(str(count))

            img = cv.imread(f"imgs/{img_file}")
            count = detect_and_count(img, s_level=0, v_level=thresh)
            counts.append(str(count))

        print(f"\rProgress: {thresh/255 * 100:.2f}%", end="")
        f.write(f"{str(thresh)};{';'.join(counts)}\n")
