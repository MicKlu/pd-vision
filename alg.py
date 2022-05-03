import cv2 as cv
import numpy as np
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
        self.__img_path = img_path
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
        self.img_original_bgr = cv.imread(self.__img_path)

class BaseBlobAlgorithm(BaseAlgorithm):

    def _segmentation(self):
        self._thresholding()
        self._morphology()
        self._extraction()

    def _thresholding(self): pass
    def _morphology(self): pass
    def _extraction(self): pass

    def _opening_closing(self, img, open_ksize: tuple = (2, 2), close_ksize: tuple = (16, 16)):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, open_ksize)
        img_opened = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, close_ksize)
        img_closed = cv.morphologyEx(img_opened, cv.MORPH_CLOSE, kernel)

        return img_closed
    
    def _closing_opening(self, img, close_ksize: tuple = (16, 16), open_ksize: tuple = (5, 5)):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, close_ksize)
        img_closed = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, open_ksize)
        img_opened = cv.morphologyEx(img_closed, cv.MORPH_OPEN, kernel)

        return img_opened

class BaseHsvBlobAlgorithm(BaseBlobAlgorithm):

    def __init__(self, img_path: str):
        super().__init__(img_path)
        self.img_prep_bgr = None
        self.img_prep_hsv = None
        self.img_prep_h, self.img_prep_s, self.img_prep_v = (None, None, None)
        self.img_h_thresh, self.img_s_thresh, self.img_v_thresh = (None, None, None)
        self.img_morphed = None
        self.blobs = None
        self.valid_blobs = None
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
        self.img_morphed = self._opening_closing(img_sv_thresh)

    def _extraction(self):
        self.blobs, hierarchy = cv.findContours(self.img_morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        # cv.drawContours(img, contours, -1, (0, 0, 255), 3)
    
    def _counting(self):
        self.count_result = 0
        self.valid_blobs = []
        for blob in self.blobs:
            if cv.contourArea(blob) > self._min_blob_size:
                self.valid_blobs.append(blob)
                self.count_result += 1

    def _bluring(self, img, median_blur_ksize: int = 3, blur_ksize: tuple = (3, 3)):
        if median_blur_ksize is not None:
            img_blur = cv.medianBlur(img, median_blur_ksize)
        if blur_ksize is not None:
            img_blur = cv.blur(img_blur, blur_ksize)
        return img_blur

    def _sharpening(self, img, kernel=None):
        if kernel is None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
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

class MedianBasedThresholdingAlgorithm(BaseHsvBlobAlgorithm):

    def _thresholding(self):
        super()._thresholding()
        (h_hist, s_hist, v_hist) = hist.get_histogram(self.img_prep_hsv, color="hsv", normalize=True)
        s_thresh, s_thresh_val = hf.median_thresh(s_hist)
        v_thresh, v_thresh_val = hf.median_thresh(v_hist)

        _, self.img_s_thresh = cv.threshold(self.img_prep_s, s_thresh, 255, cv.THRESH_BINARY)
        _, self.img_v_thresh = cv.threshold(self.img_prep_v, v_thresh, 255, cv.THRESH_BINARY_INV)
        self.img_h_thresh = np.full_like(self.img_prep_h, 255)

