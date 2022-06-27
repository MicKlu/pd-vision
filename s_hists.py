#!/usr/bin/python3

from alg.custom import CustomHsvBlobAlgorithm
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
from hist import get_single_channel_histogram
from scipy import optimize

def gauss(x, a, mu, sigma):
    return a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

def fit_gauss(x, y):
    peak = x[y > np.exp(-0.5) * y.max()]
    sigma0 = 0.5*(peak.max() - peak.min())

    popt, pcov = optimize.curve_fit(gauss, x, y, (np.max(y), 127, sigma0))
    return popt

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
        plt.imshow(alg.img_s_h_masked, cmap="gray")
        plt.axis("off")

        plt.subplot(sp_dim[0], sp_dim[1], 2 * img_index + 2)
        plt.hist(alg.img_s_h_masked.ravel(), 256, [1,256])

        s_hist = get_single_channel_histogram(alg.img_s_h_masked)[1:]

        x = np.arange(1, 256)
        (a, mu, sigma) = fit_gauss(x, s_hist)
        
        x = np.linspace(x[0], x[-1], x.size * 10)
        plt.plot(x, gauss(x, a, mu, sigma))

        print(mu) # mu
        print(sigma) # sigma

    plt.show()