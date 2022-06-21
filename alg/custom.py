from . import *
from alg.ref import ReferenceAlgorithm
import hist_filter as hf

class CustomHsvBlobAlgorithm(ReferenceAlgorithm):

    def _preprocessing(self):
        super()._preprocessing()

        img_hsv = cv.cvtColor(self.img_prep_bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(img_hsv)

        self.img_prep_h_uneq = h
        self.img_prep_s_uneq = s
        self.img_prep_v_uneq = v

        # s_norm = cv.equalizeHist(s)
        v_norm = cv.equalizeHist(v)

        # s_norm = cv.morphologyEx(s_norm, cv.MORPH_TOPHAT, (5, 5))
        v_norm = cv.morphologyEx(v_norm, cv.MORPH_TOPHAT, (5, 5))

        self.img_prep_h = h
        self.img_prep_s = s
        self.img_prep_v = v_norm

        self.img_prep_hsv = cv.merge([h, s, v_norm])
        self.img_prep_bgr = cv.cvtColor(self.img_prep_hsv, cv.COLOR_HSV2BGR)

        # Apply preprocessing to all spaces (HSV -> BRG -> HSV)
        # hsv2bgr2hsv = self.img_prep_hsv = cv.cvtColor(self.img_prep_bgr, cv.COLOR_BGR2HSV)
        # self.img_prep_h, self.img_prep_s, self.img_prep_v = self.img_prep_h, self.img_prep_s, self.img_prep_v = cv.split(hsv2bgr2hsv)

    def _sharpening(self, img, kernel=None):
        return super()._sharpening(img, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))

    def _thresholding(self):
        self._color_reduction()

        self.img_s_h_masked = cv.bitwise_and(self.img_prep_s, self.h_mask)
        self.img_v_h_masked = cv.bitwise_and(cv.bitwise_not(self.img_prep_v), self.h_mask)

        # to be continued

        # BEGIN for now
        self.img_h_thresh = self.h_mask
        self.img_s_thresh = self.img_s_h_masked
        self.img_v_thresh = self.img_v_h_masked
        # END for now

    def _color_reduction(self):
        h_uneq = np.copy(self.img_prep_h_uneq)

        # Move all zeros to 179 (almost same red color)
        _, m = cv.threshold(h_uneq, 0, 179, cv.THRESH_BINARY_INV)
        h_uneq = cv.bitwise_or(h_uneq, m)

        # Remove most significant color
        h_uneq_threshed = self._reduce_colors(h_uneq)

        for i in range(0, 3):   # max 5; optimal 3
            h_uneq_threshed = self._reduce_colors(h_uneq_threshed)

        h_uneq_threshed = self._reduce_colors(h_uneq_threshed, close=True)

        self.h_mask = self._fill_holes(h_uneq, h_uneq_threshed)

    def _reduce_colors(self, img_h, close=False):
        img_hist = self._get_h_histogram(img_h)[1:]

        pix_val = img_hist.tolist().index(img_hist.max()) + 1
        mask = np.where(img_h == pix_val, 0, 255).astype("uint8")
        print(f"pix_val: {pix_val}")
        print(np.sum(mask == 0))

        img_thresh = cv.bitwise_and(img_h, mask)

        if close:
            _, mask = cv.threshold(img_thresh, 0, 255, cv.THRESH_BINARY)
            mask = self._closing_opening(mask, close_ksize=(3, 3), open_ksize=(3, 3))
            img_thresh = cv.bitwise_and(img_h, mask)

        return img_thresh

    def _fill_holes(self, img_original, img_hollow):
        _, mask = cv.threshold(img_hollow, 0, 255, cv.THRESH_BINARY)
        cont, hier = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        for c in cont:
            cv.drawContours(mask, [c], 0, 255, -1)

        return cv.bitwise_and(img_original, mask)

    def _get_single_channel_histogram(self, channel, bins=256):
        hist = np.histogram(channel.ravel(), bins, [0,bins])
        hist = hist[0] / hist[0].max()
        return hist

    def _get_h_histogram(self, h_channel):
        return self._get_single_channel_histogram(h_channel, 180)