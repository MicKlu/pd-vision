import cv2 as cv
import numpy as np
import math
import hist
import hist_filter as hf

class IAlgorithm:
    def _capture(self): pass
    def _preprocessing(self): pass
    def _segmentation(self): pass
    def _counting(self): pass
    def count(self) -> int: pass

class BaseAlgorithm(IAlgorithm):
    
    def __init__(self, img_path: str):
        self._img_path = img_path
        self.img_original_bgr = None
        self.count_result = None

    def count(self) -> int:
        if self.count_result is not None:
            return self.count_result

        self._capture()
        self._preprocessing()
        self._segmentation()
        self._counting()
        return self.count_result

    def _capture(self):
        self.img_original_bgr = cv.imread(self._img_path)

        w = np.size(self.img_original_bgr, 1)
        h = np.size(self.img_original_bgr, 0)
        target_d = 1920 * 1080
        ratio = w / h

        target_h = math.sqrt(target_d * h / w)
        target_w = target_h * ratio

        target_h = math.ceil(target_h)
        target_w = math.ceil(target_w)

        self.img_original_bgr = cv.resize(self.img_original_bgr, (target_w, target_h))

class BaseBlobAlgorithm(BaseAlgorithm):

    def _segmentation(self):
        self._thresholding()
        self._morphology()
        self._extraction()

    def _thresholding(self): pass
    def _morphology(self): pass
    def _extraction(self): pass

    def _opening_closing(self, img, open_ksize: tuple = (3, 3), close_ksize: tuple = (17, 17), open_iterations=1, close_iterations=1):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, open_ksize)
        img_opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=open_iterations)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, close_ksize)
        img_closed = cv.morphologyEx(img_opened, cv.MORPH_CLOSE, kernel, iterations=close_iterations)

        return img_closed
    
    def _closing_opening(self, img, close_ksize: tuple = (16, 16), open_ksize: tuple = (5, 5), close_iterations=1, open_iterations=1):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, close_ksize)
        img_closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=close_iterations)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, open_ksize)
        img_opened = cv.morphologyEx(img_closed, cv.MORPH_OPEN, kernel, iterations=open_iterations)

        return img_opened

class BaseHsvBlobAlgorithm(BaseBlobAlgorithm):

    def __init__(self, img_path: str):
        super().__init__(img_path)
        self.img_prep_bgr = None
        self.img_prep_hsv = None
        self.img_prep_h, self.img_prep_s, self.img_prep_v = (None, None, None)
        self.img_h_thresh, self.img_s_thresh, self.img_v_thresh = (None, None, None)
        self.img_bitwise = None
        self.img_morphed = None
        self.blobs = None
        self.valid_blobs = None

        self._label_connectivity = 8
        self._min_blob_size = 500

    def _preprocessing(self):
        self.img_prep_bgr = np.copy(self.img_original_bgr)

        self.img_prep_bgr = self._bluring(self.img_prep_bgr)
        self.img_prep_bgr = self._sharpening(self.img_prep_bgr)

    def _thresholding(self):
        self.img_prep_hsv = cv.cvtColor(self.img_prep_bgr, cv.COLOR_BGR2HSV)
        self.img_prep_h, self.img_prep_s, self.img_prep_v = cv.split(self.img_prep_hsv)

    def _morphology(self):
        img_sv_thresh = cv.bitwise_and(self.img_s_thresh, self.img_v_thresh)
        self.img_bitwise = img_sv_thresh
        self.img_morphed = self._opening_closing(img_sv_thresh)

    def _extraction(self):
        _, _, stats, centroids = cv.connectedComponentsWithStats(self.img_morphed, connectivity=self._label_connectivity)
        self.labels = [{ "stats": stats[i], "centroids": centroids[i]} for i in range(1, len(stats))]
        self.blobs, hierarchy = cv.findContours(self.img_morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    def _counting(self):
        self.count_result = 0
        self.valid_blobs = []

        for label in self.labels:
            size = label["stats"][cv.CC_STAT_AREA]
            if size >= self._min_blob_size:
                self.valid_blobs.append(label)
                self.count_result += 1

    def _bluring(self, img, median_blur_ksize: int = 3, blur_ksize: tuple = (3, 3)):
        if median_blur_ksize is not None:
            img_blur = cv.medianBlur(img, median_blur_ksize)
        if blur_ksize is not None:
            img_blur = cv.blur(img_blur, blur_ksize)
        return img_blur

    def _sharpening(self, img, kernel=None):
        if kernel is None:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_sharp = cv.filter2D(img, -1, kernel)
        return img_sharp

class ReferenceAlgorithm(BaseHsvBlobAlgorithm):
    
    def __init__(self, img_path: str, s_thresh_level=23, v_thresh_level=96):
        super().__init__(img_path)
        self.s_thresh_level = s_thresh_level
        self.v_thresh_level = v_thresh_level

    def _thresholding(self):
        super()._thresholding()
        _, self.img_s_thresh = cv.threshold(self.img_prep_s, self.s_thresh_level, 255, cv.THRESH_BINARY)
        _, self.img_v_thresh = cv.threshold(self.img_prep_v, self.v_thresh_level, 255, cv.THRESH_BINARY_INV)
        self.img_h_thresh = np.full_like(self.img_prep_h, 255)

class ReferenceAlgorithmToleranceCount(ReferenceAlgorithm):

    def __init__(self, img_path: str, s_thresh_level=23, v_thresh_level=96, tolerance=900):
        super().__init__(img_path, s_thresh_level, v_thresh_level)
        self.tolerance = tolerance

    def _counting(self):
        minimum = 0
        maximum = np.size(self.img_morphed)
        
        L = len(self.labels)

        prev_average = -1

        while True:
            sizes = []

            for i in range(0, L):
                size = self.labels[i]["stats"][cv.CC_STAT_AREA]
                if size >= minimum and size <= maximum:
                    sizes.append(size)

            average = np.average(sizes)

            if average == prev_average:
                break

            std = np.std(sizes)
            variation = (maximum - minimum) / average * 100

            # print(average)
            # print(std)
            # print(variation)
            # print()

            if variation <= self.tolerance:
                break
            else:
                extreme_average = (np.max(sizes) + np.min(sizes)) / 2
                if extreme_average > average:
                    maximum = average + std
                    minimum = np.min(sizes)
                elif extreme_average < average:
                    minimum = average - std
                    maximum = np.max(sizes)
            
            prev_average = average

        cutoff_low = (1 - self.tolerance / 1000) * average
        cutoff_high = (1 + self.tolerance / 1000) * average

        # print(cutoff_low)
        # print(cutoff_high)

        self.count_result = 0
        self.valid_blobs = []

        for label in self.labels:
            size = label["stats"][cv.CC_STAT_AREA]
            if size >= cutoff_low and size <= cutoff_high:
                self.valid_blobs.append(label)
                self.count_result += 1

class ReferenceAlgorithmWithMorphKernels(ReferenceAlgorithm):
    
    def __init__(self, img_path: str, s_thresh_level, v_thresh_level, open_ksize, close_ksize):
        super().__init__(img_path, s_thresh_level, v_thresh_level)
        self.open_ksize = open_ksize
        self.close_ksize = close_ksize

    def _morphology(self):
        img_sv_thresh = cv.bitwise_and(self.img_s_thresh, self.img_v_thresh)
        self.img_bitwise = img_sv_thresh
        self.img_morphed = self._opening_closing(img_sv_thresh, open_ksize=self.open_ksize, close_ksize=self.close_ksize)

class MedianBasedThresholdingAlgorithm(BaseHsvBlobAlgorithm):

    def _thresholding(self):
        super()._thresholding()
        (h_hist, s_hist, v_hist) = hist.get_histogram(self.img_prep_hsv, color="hsv", normalize=True)
        s_thresh, _ = hf.median_thresh(s_hist)
        v_thresh, _ = hf.median_thresh(v_hist)

        _, self.img_s_thresh = cv.threshold(self.img_prep_s, s_thresh, 255, cv.THRESH_BINARY)
        _, self.img_v_thresh = cv.threshold(self.img_prep_v, v_thresh, 255, cv.THRESH_BINARY_INV)
        self.img_h_thresh = np.full_like(self.img_prep_h, 255)

class MedianBasedFilteringAlgorithm(BaseHsvBlobAlgorithm):

    def _thresholding(self):
        super()._thresholding()
        (h_hist, s_hist, v_hist) = hist.get_histogram(self.img_prep_hsv, color="hsv", normalize=True)
        _, s_thresh_val = hf.median_thresh(s_hist)
        _, v_thresh_val = hf.median_thresh(v_hist)

        s_filter = []
        v_filter = []

        for pix_val in range(0, 256):
            if s_hist[0][pix_val] >= s_thresh_val:
                s_filter.append(pix_val)

            if v_hist[0][pix_val] >= v_thresh_val:
                v_filter.append(pix_val)

        self.img_s_thresh = np.where(np.isin(self.img_prep_s, s_filter), 0, 255).astype("uint8")
        self.img_v_thresh = np.where(np.isin(self.img_prep_v, v_filter), 0, 255).astype("uint8")
        self.img_h_thresh = np.full_like(self.img_prep_h, 255)

class MedianBasedThresholdingORAlgorithm(MedianBasedThresholdingAlgorithm):

    def _morphology(self):
        img_s_thresh_morphed = self._opening_closing(self.img_s_thresh, open_ksize=(3, 3), open_iterations=3)
        img_v_thresh_morphed = self._opening_closing(self.img_v_thresh, open_ksize=(3, 3), open_iterations=3)
        self.img_morphed = cv.bitwise_or(img_s_thresh_morphed, img_v_thresh_morphed)
        self.img_bitwise = self.img_morphed

class MedianBasedFilteringORAlgorithm(MedianBasedFilteringAlgorithm):

    def _morphology(self):
        img_s_thresh_morphed = self._opening_closing(self.img_s_thresh, open_ksize=(3, 3), open_iterations=3)
        img_v_thresh_morphed = self._opening_closing(self.img_v_thresh, open_ksize=(3, 3), open_iterations=3)
        self.img_morphed = cv.bitwise_or(img_s_thresh_morphed, img_v_thresh_morphed)
        self.img_bitwise = self.img_morphed

class CustomHsvBlobAlgorithm(ReferenceAlgorithm):

    def _preprocessing(self):
        super()._preprocessing()

        img_hsv = cv.cvtColor(self.img_prep_bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(img_hsv)

        self.img_prep_h_uneq = h
        self.img_prep_s_uneq = s
        self.img_prep_v_uneq = v

        v_norm = cv.equalizeHist(v)

        # cv.namedWindow("dsa", cv.WINDOW_NORMAL)
        # cv.imshow("dsa", s)

        # cv.waitKey()

        self.img_prep_bgr = cv.cvtColor(cv.merge([h, s, v_norm]), cv.COLOR_HSV2BGR)

        # img_hsv = cv.cvtColor(self.img_prep_bgr, cv.COLOR_BGR2HSV)
        # h,s,v = cv.split(img_hsv)

        # cv.namedWindow("cxz", cv.WINDOW_NORMAL)
        # cv.imshow("cxz", s)

        # cv.waitKey()

    def _sharpening(self, img, kernel=None):
        return super()._sharpening(img, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))