#!/usr/bin/python3

import sys

sys.path.append("..")

import cv2 as cv
from alg import ReferenceAlgorithm
import os

for img_file in ["img0.jpg", "img1.jpg", "img2.jpg"]:
    print(img_file)
    with open(f"_calc_tresh_results_moderate_samples_{img_file}.csv", "w") as f:
        f.write("s_thresh;v_thresh;count\n")
        
        for s_thresh in range(0, 256):
            for v_thresh in range(0, 256):
                a = ReferenceAlgorithm(f"../imgs/moderate_samples/{img_file}", s_thresh,v_thresh)
                a.count()
                f.write(f"{s_thresh};{v_thresh};{len(a.blobs)}\n")

                print(f"\rProgress: {(s_thresh * 256 + v_thresh)/256/256 * 100:.2f}%", end="")
        print()
