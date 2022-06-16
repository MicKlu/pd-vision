import hist
import hist_filter as hf

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