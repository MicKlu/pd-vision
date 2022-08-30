from . import *

class ReferenceAlgorithm(BaseHsvBlobAlgorithm):
    
    def __init__(self, img_path: str, s_thresh_level=23, v_thresh_level=96):
        super().__init__(img_path)
        self.s_thresh_level = s_thresh_level
        self.v_thresh_level = v_thresh_level

    @debug_time("execution_time_thresholding")
    def _thresholding(self):
        super()._thresholding()
        _, self.img_s_thresh = cv.threshold(self.img_prep_s, self.s_thresh_level, 255, cv.THRESH_BINARY)
        _, self.img_v_thresh = cv.threshold(self.img_prep_v, self.v_thresh_level, 255, cv.THRESH_BINARY_INV)
        self.img_h_thresh = np.full_like(self.img_prep_h, 255)

class ReferenceAlgorithmToleranceCount(ReferenceAlgorithm):

    def __init__(self, img_path: str, s_thresh_level=23, v_thresh_level=96, tolerance=900):
        super().__init__(img_path, s_thresh_level, v_thresh_level)
        self.tolerance = tolerance

    @debug_time("execution_time_counting")
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