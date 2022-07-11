#!/usr/bin/python3

from alg.custom import CustomHsvBlobAlgorithm
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np

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
        plt.imshow(alg.img_s_morphed, cmap="gray")
        plt.axis("off")

        plt.subplot(sp_dim[0], sp_dim[1], 2 * img_index + 2)
        plt.imshow(alg.img_v_morphed, cmap="gray")
        plt.axis("off")

    plt.show()