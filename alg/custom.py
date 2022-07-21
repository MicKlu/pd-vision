from . import *
from alg.ref import ReferenceAlgorithm
from scipy import optimize, signal

class CustomHsvBlobAlgorithm(ReferenceAlgorithm):

    def __init__(self, img_path):
        super().__init__(img_path)

        self.min_blob_size = math.ceil(math.pi * 7**2)
        self.safe_area = 0.8

    def _preprocessing(self):
        super()._preprocessing()

        img_hsv = cv.cvtColor(self.img_prep_bgr, cv.COLOR_BGR2HSV)
        h, s, v = cv.split(img_hsv)

        self.img_prep_h_uneq = h
        self.img_prep_s_uneq = s
        self.img_prep_v_uneq = v

        v_norm = cv.equalizeHist(v)

        # v_norm = cv.morphologyEx(v_norm, cv.MORPH_TOPHAT, (5, 5))
        # kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (25, 25))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
        v_norm = cv.morphologyEx(v_norm, cv.MORPH_TOPHAT, kernel)

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

        self.img_v_h_masked = cv.GaussianBlur(self.img_v_h_masked, (3, 3), 0)

        s_hist = self._get_single_channel_histogram(self.img_s_h_masked)[1:]
        v_hist = self._get_single_channel_histogram(self.img_v_h_masked)[1:]

        x = np.arange(1, 256)
        (a, mu, sigma) = self._fit_gauss(x, s_hist)
        s_pix_val = np.ceil(mu + 1.5 * sigma)

        # peaks, _ = signal.find_peaks(np.append(v_hist, 0), np.average(v_hist))
        # v_pix_val = np.max(peaks)

        # v_pix_val = v_hist.tolist().index(v_hist.max()) + 0

        v_pix_val = 254

        print(s_pix_val)
        print(v_pix_val)

        _, self.img_s_thresh = cv.threshold(self.img_s_h_masked, s_pix_val, 255, cv.THRESH_BINARY)
        _, self.img_v_thresh = cv.threshold(self.img_v_h_masked, v_pix_val, 255, cv.THRESH_BINARY)

        self.img_h_thresh = self.h_mask

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

        mask = self._fill_holes(h_uneq_threshed)
        h_uneq_threshed = cv.bitwise_and(h_uneq, mask)

        _, self.h_mask = cv.threshold(h_uneq_threshed, 1, 255, cv.THRESH_BINARY)

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

    def _fill_holes(self, img_hollow):
        _, mask = cv.threshold(img_hollow, 0, 255, cv.THRESH_BINARY)
        cont, hier = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

        img_filled = mask
        for c in cont:
            cv.drawContours(img_filled, [c], 0, 255, -1)

        return img_filled

    def _get_single_channel_histogram(self, channel, bins=256):
        hist = np.histogram(channel.ravel(), bins, [0,bins])
        hist = hist[0] / hist[0].max()
        return hist

    def _get_h_histogram(self, h_channel):
        return self._get_single_channel_histogram(h_channel, 180)

    def _fit_gauss(self, x, y):
        peak = x[y > np.exp(-0.5) * y.max()]
        sigma0 = 0.5*(peak.max() - peak.min())

        popt, pcov = optimize.curve_fit(self.__gauss, x, y, (np.max(y), 127, sigma0))
        return popt

    def __gauss(self, x, a, mu, sigma):
        return a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu)**2 / (2 * sigma**2))

    def _morphology(self):
        self.img_bitwise = cv.bitwise_or(self.img_s_thresh, self.img_v_thresh)
        sv_blobs = self._separate(self.img_bitwise)

        img_sv_morphed_big = np.copy(sv_blobs[0])

        img_sv_morphed_big = self._closing_with_filling(img_sv_morphed_big, (17, 17))
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
        img_sv_morphed_big = cv.morphologyEx(img_sv_morphed_big, cv.MORPH_OPEN, kernel, iterations=1)

        img_sv_morphed_big = self._remove_border_blobs(img_sv_morphed_big)

        while True:
            img_separated = self._separate_blobs(img_sv_morphed_big)
            if np.all(img_separated == img_sv_morphed_big):
                img_sv_morphed_big = img_separated
                break
            img_sv_morphed_big = img_separated

        self.img_morphed = img_sv_morphed_big

    def _separate(self, img):
        contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
        areas = np.array([ cv.contourArea(c) for c in contours ])
        avg = np.average(areas)
        
        print(areas.min())
        print(areas.max())
        print(avg)

        big_blobs = np.zeros_like(img)
        for c in contours:
            if cv.contourArea(c) > avg:
                cv.drawContours(big_blobs, [c], 0, 255, -1)

        small_blobs = img - big_blobs

        return (big_blobs, small_blobs)

    def _closing_with_filling(self, img, ksize=(3, 3)):
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
        img = cv.morphologyEx(img, cv.MORPH_DILATE, kernel)
        img = self._fill_holes(img)
        return cv.morphologyEx(img, cv.MORPH_ERODE, kernel)

    def _remove_border_blobs(self, img_morphed):
        blobs, _ = cv.findContours(img_morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        img_height = np.size(img_morphed, 0)
        img_width = np.size(img_morphed, 1)
        offset_v = (1 - self.safe_area) * img_height / 2
        offset_h = (1 - self.safe_area) * img_width / 2

        for blob in blobs:
            moments = cv.moments(blob)

            cx = int(moments['m10']/moments['m00'])
            cy = int(moments['m01']/moments['m00'])

            if not ((cx < offset_h or cx > img_width - offset_h)
                    or (cy < offset_v or cy > img_height - offset_v)):
                continue

            cv.drawContours(img_morphed, [blob], 0, 0, -1)

        return img_morphed

    def _separate_blobs(self, img_morphed):
        blobs, _ = cv.findContours(img_morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for blob in blobs:
            hull = cv.convexHull(blob)

            blob_area = float(cv.contourArea(blob))
            hull_area = cv.contourArea(hull)

            if hull_area == 0:
                solidity = 1
            else:
                solidity = blob_area / hull_area

            if solidity > 0.75:
                continue

            leftmost = tuple(blob[blob[:,:,0].argmin()][0])
            rightmost = tuple(blob[blob[:,:,0].argmax()][0])
            topmost = tuple(blob[blob[:,:,1].argmin()][0])
            bottommost = tuple(blob[blob[:,:,1].argmax()][0])

            defect_points = self._get_convexity_defect_points(blob)

            if len(defect_points) < 2:
                continue

            blob_canvas = np.zeros_like(img_morphed)
            cv.drawContours(blob_canvas, [blob], 0, 255, -1)
            cv.drawContours(img_morphed, [blob], 0, 0, -1)

            blob_canvas_bgr = cv.cvtColor(blob_canvas, cv.COLOR_GRAY2BGR)

            for defect in defect_points:
                cv.putText(blob_canvas_bgr, f"{defect['distance']}", (defect['coords'][0], defect['coords'][1] - 5), cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, [255, 0, 0], 1, cv.LINE_AA)
                cv.circle(blob_canvas_bgr, defect['coords'], 2, [0,0,255], -1)

                closest_defect = None
                min_d = np.inf
                for other_defect in defect_points:
                    if other_defect is defect:
                        continue

                    ax, ay = defect['coords']
                    bx, by = other_defect['coords']

                    d = math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)
                    if d < min_d:
                        min_d = d
                        closest_defect = other_defect

                if not closest_defect:
                    continue

                cv.line(blob_canvas_bgr, defect['coords'], closest_defect['coords'], [0, 0, 0])
                cv.line(blob_canvas, defect['coords'], closest_defect['coords'], [0, 0, 0])

            blob_roi = blob_canvas[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
            blob_roi_bgr = blob_canvas_bgr[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]

            img_canvas = np.copy(self.img_original_bgr)
            cv.drawContours(img_canvas, [blob], 0, [0, 0, 255], 3)
            img_roi = img_canvas[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]

            cv.imshow("dsa", blob_roi_bgr)
            cv.imshow("ewq", img_roi)

            print()
            print(f"Area: {blob_area}")
            print(f"Solidity: {solidity}")
            print()

            cv.waitKey()

            ksize = (7, 7)
            if blob_area < 1000:
                ksize = (5, 5)
            if blob_area < 500:
                ksize = (3, 3)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
            blob_canvas = cv.morphologyEx(blob_canvas, cv.MORPH_ERODE, kernel, iterations=1)
            blob_canvas_bgr = cv.morphologyEx(blob_canvas_bgr, cv.MORPH_ERODE, kernel, iterations=1)

            img_morphed = cv.bitwise_or(img_morphed, blob_canvas)

            blob_roi = blob_canvas[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]
            blob_roi_bgr = blob_canvas_bgr[topmost[1]:bottommost[1], leftmost[0]:rightmost[0]]

            new_blobs, _ = cv.findContours(blob_roi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

            for new_blob in new_blobs:
                new_hull = cv.convexHull(new_blob)
                new_blob_area = float(cv.contourArea(new_blob))
                new_hull_area = cv.contourArea(new_hull)

                if new_hull_area == 0:
                    new_solidity = 1
                else:
                    new_solidity = new_blob_area / new_hull_area

                cv.drawContours(blob_roi_bgr, [new_hull], 0, (0, 255, 0), 1)

                new_leftmost = tuple(new_blob[new_blob[:,:,0].argmin()][0])
                new_topmost = tuple(new_blob[new_blob[:,:,1].argmin()][0])

                cv.putText(blob_roi_bgr, f"{new_solidity:.3f}", (new_leftmost[0], new_topmost[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)
                print(new_solidity)

            cv.imshow("dsa", blob_roi_bgr)

            cv.waitKey()

        return img_morphed

    def _get_convexity_defect_points(self, blob):
        defect_points = []

        hull_indices = cv.convexHull(blob, returnPoints=False)
        try:
            defects = cv.convexityDefects(blob, hull_indices)

            for defect in defects:
                _, _, far_idx, d = defect[0]
                far = blob[far_idx][0]
                if d < 3000:
                    continue
                defect_points.append({
                    'coords': far,
                    'distance': d
                })
        finally:
            return defect_points