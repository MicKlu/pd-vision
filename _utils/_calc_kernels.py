#!/usr/bin/python3

import sys

sys.path.append("..")

import cv2 as cv
from alg import ReferenceAlgorithmWithMorphKernels
import os
import numpy as np

img_file = "img0.jpg"

lines = 625
objects = 19

print(f"s_thresh\tv_thresh\tok\tck\tprogress")

with open(f"_calc_tresh_results_easy_samples_{img_file}.csv", "r") as f:
    with open(f"_calc_kernels_out/_calc_kernels_results_easy_samples_{img_file}.csv", "w") as fout:
        
        f.readline()

        k_headers = []
        for ok in range(3, 13, 2):
            for ck in range(3, 52, 2):
                k_headers.append(f"{ok}/{ck}")

        fout.write(f"s_thresh,v_thresh,{','.join(k_headers)}\n")
        
        line_num = 1
        while True:
            progress = line_num / lines

            l: str = f.readline()
            
            if not l:
                break

            l = l[:-1].split(",")

            s_thresh = int(l[0])
            v_thresh = int(l[1])
            count = int(l[2])

            k_counts = []

            for ok in range(3, 13, 2):
                for ck in range(3, 52, 2):
                    a = ReferenceAlgorithmWithMorphKernels(f"../imgs/easy_samples/{img_file}", s_thresh, v_thresh, (ok, ok), (ck, ck))
                    a.count()

                    count = len(a.blobs)

                    k_counts.append(str(count))
                    
                    if(count == objects):
                        cv.imwrite(f"_calc_kernels_out/{s_thresh},{v_thresh}_{ok},{ck}_{img_file}.png", a.img_morphed)
                    
                    print(f"{s_thresh:8}\t{v_thresh:8}\t{ok:2}\t{ck:2}\t{progress:7.2f}%\r", end="")

            fout.write(f"{s_thresh},{v_thresh},{','.join(k_counts)}\n")

            line_num += 1

        print()