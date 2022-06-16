#!/usr/bin/python3

import sys

sys.path.append("..")

import cv2 as cv
import hist
import os
import numpy as np

imgs_list = os.listdir("imgs")
s_threshs = [38, 86, 39, 159, 40, 65, 64, 30, 77, 78, 39, 52, 123, 79]
v_threshs = [104, 156, 100, 72, 93, 64, 37, 37, 36, 36, 99, 94, 102, 31]

with open("_calc_hist_thresh_dep.csv", "w") as f:
# with open("temp.tmp", "w") as f:
    f.write("img;s_avg;s_med;s_max;s_thresh_val;s_thresh;v_avg;v_med;v_max;v_thresh_val;v_thresh\n")
    for (i, img_file) in enumerate(imgs_list):
        img = cv.imread(f"imgs/{img_file}")
        h_hist, s_hist, v_hist = hist.get_histogram(img, normalize=True)

        f.write(f"{img_file};")
        f.write(f"{np.average(s_hist[0])};")
        f.write(f"{np.median(s_hist[0])};")
        f.write(f"{np.max(s_hist[0])};")
        f.write(f"{s_hist[0][s_threshs[i]]};")
        f.write(f"{s_threshs[i]};")

        f.write(f"{np.average(v_hist[0])};")
        f.write(f"{np.median(v_hist[0])};")
        f.write(f"{np.max(v_hist[0])};")
        f.write(f"{v_hist[0][v_threshs[i]]};")
        f.write(f"{v_threshs[i]}")
        f.write("\n")
