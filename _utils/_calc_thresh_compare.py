#!/usr/bin/python3

import sys

sys.path.append("..")

import cv2 as cv
from alg import ReferenceAlgorithm
import os
import numpy as np

img_difficulty = "moderate"
img_file = "img0.jpg"

for img_file in ["img0.jpg", "img1.jpg", "img2.jpg"]:

    img_prev = None
    count_prev = None

    with open(f"_calc_tresh_results_{img_difficulty}_samples_{img_file}.csv", "r") as f:
        with open(f"_calc_thresh_compare_out/_calc_tresh_results_{img_difficulty}_samples_{img_file}_compared.csv", "w") as fout:
            fout.write(f"{f.readline()}")
            while True:
                l: str = f.readline()
                
                if not l:
                    break

                l = l[:-1].split(",")

                s_thresh = int(l[0])
                v_thresh = int(l[1])
                count = int(l[2])

                a = ReferenceAlgorithm(f"../imgs/{img_difficulty}_samples/{img_file}", s_thresh, v_thresh)
                a.count()

                save = False

                if count == count_prev and img_prev is not None:
                    if not np.all(img_prev == a.img_morphed):
                        save = True
                else:
                    save = True

                if save:
                    fout.write(f"{s_thresh},{v_thresh},{count}\n")
                    cv.imwrite(f"_calc_thresh_compare_out/{s_thresh},{v_thresh}_{img_difficulty}_{img_file}.png", a.img_morphed)

                print(f"{s_thresh:3},{v_thresh:3},{save}\t\r", end="")

                img_prev = a.img_morphed
                count_prev = count

            print()
