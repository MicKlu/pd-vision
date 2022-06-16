import hist
import hist_filter as hf

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