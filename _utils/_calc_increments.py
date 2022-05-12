#!/usr/bin/python3

import sys
import os

sys.path.append(os.getcwd())

import cv2 as cv
import hist
import alg as a
import numpy as np

class DumbAlgorithm(a.BaseHsvBlobAlgorithm):

    def __init__(self, img_path, fd):
        super().__init__(img_path)
        self.__fd = fd

    def _thresholding(self):
        super()._thresholding()

        (h_hist, s_hist, v_hist) = hist.get_histogram(self.img_prep_hsv, color="hsv", normalize=True)

        d_s_hist = s_hist[0][1:] - s_hist[0][:-1]
        d_v_hist = v_hist[0][1:] - v_hist[0][:-1]

        for start_pix_val in range(0, 255 - 10):
            s_range = d_s_hist[start_pix_val:start_pix_val + 10]
            v_range = d_v_hist[start_pix_val:start_pix_val + 10]

            np.set_printoptions(linewidth=np.inf)

            self.__fd.write(f'{self._img_path};{start_pix_val}-{start_pix_val+10};{np.array2string(s_range)};{np.average(s_range)};{np.array2string(v_range)};{np.average(v_range)}\n')


    
    def _morphology(self): pass
    def _extraction(self): pass
    def _counting(self): pass

if __name__ == "__main__":
    ### Load image
    imgs_list = os.listdir("imgs")

    with open("_calc_increments.csv", "w") as f:
        for img_name in imgs_list:
            print(f"imgs/{img_name}")
            alg = DumbAlgorithm(f"imgs/{img_name}", f)
            alg.count()