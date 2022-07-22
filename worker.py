import cv2 as cv
import numpy as np

from PyQt5.Qt import Qt
from PyQt5.Qt import QImage, QPixmap

from alg.ref import ReferenceAlgorithm
from alg.custom import CustomHsvBlobAlgorithm

class CountingWorkerError(Exception):

    def __init__(self, args):
        super().__init__(args)

class CountingWorker:

    ALGORITHM_REF = 0
    ALGORITHM_CUSTOM = 1

    IMAGE_ORIGINAL = 0
    IMAGE_PREPROCESSED = 1
    IMAGE_PREPROCESSED_H = 2
    IMAGE_PREPROCESSED_S = 3
    IMAGE_PREPROCESSED_V = 4
    IMAGE_THRESHOLD_H = 5
    IMAGE_THRESHOLD_S = 6
    IMAGE_THRESHOLD_V = 7
    IMAGE_MORPHED = 8
    IMAGE_COUNTING = 9

    def __init__(self, window: 'MainWindow', img_path: str):
        self.__window = window
        self.__img_path = img_path
        self.__alg = None
        self.__alg_id = None

        img = cv.imread(img_path)
        if img is None:
            raise CountingWorkerError("Plik nie może być otwarty")

        pixmap = self.__opencv2pixmap(img)
        window.imagePreviewLeft.setImage(pixmap)

    def count(self):
        self.__alg_id = self.__window.algorithmCombo.currentIndex()
        
        if self.__alg_id == CountingWorker.ALGORITHM_REF:
            self.__setup_ref_algorithm()
        elif self.__alg_id == CountingWorker.ALGORITHM_CUSTOM:
            self.__setup_custom_algorithm()

        e1 = cv.getTickCount()
        
        count = self.__alg.count()
        
        e2 = cv.getTickCount()
        time = (e2 - e1)/ cv.getTickFrequency()

        self.__window.detectedValueLabel.setText(f"{count}")
        self.__window.executionTimeValueLabel.setText(f"{time} s")

        self.__window.update_image_left()
        self.__window.update_image_right()

    def get_image(self, image_index: int):
        pixmap = QPixmap()

        if self.__alg is None:
            return pixmap

        if image_index == CountingWorker.IMAGE_ORIGINAL:
            pixmap = self.__opencv2pixmap(self.__alg.img_original_bgr)
        elif image_index == CountingWorker.IMAGE_PREPROCESSED:
            pixmap = self.__opencv2pixmap(self.__alg.img_prep_bgr)
        elif image_index == CountingWorker.IMAGE_PREPROCESSED_H:
            pixmap = self.__opencv2pixmap(self.__alg.img_prep_h, QImage.Format_Grayscale8)
        elif image_index == CountingWorker.IMAGE_PREPROCESSED_S:
            pixmap = self.__opencv2pixmap(self.__alg.img_prep_s, QImage.Format_Grayscale8)
        elif image_index == CountingWorker.IMAGE_PREPROCESSED_V:
            pixmap = self.__opencv2pixmap(self.__alg.img_prep_v, QImage.Format_Grayscale8)
        elif image_index == CountingWorker.IMAGE_THRESHOLD_H:
            pixmap = self.__opencv2pixmap(self.__alg.img_h_thresh, QImage.Format_Grayscale8)
        elif image_index == CountingWorker.IMAGE_THRESHOLD_S:
            pixmap = self.__opencv2pixmap(self.__alg.img_s_thresh, QImage.Format_Grayscale8)
        elif image_index == CountingWorker.IMAGE_THRESHOLD_V:
            pixmap = self.__opencv2pixmap(self.__alg.img_v_thresh, QImage.Format_Grayscale8)
        elif image_index == CountingWorker.IMAGE_MORPHED:
            pixmap = self.__opencv2pixmap(self.__alg.img_morphed, QImage.Format_Grayscale8)
        elif image_index == CountingWorker.IMAGE_COUNTING:
            img_output = np.copy(self.__alg.img_original_bgr)
            cv.drawContours(img_output, self.__alg.blobs, -1, (0, 0, 255), 3)

            for l in self.__alg.valid_blobs:
                pt1 = (l["stats"][cv.CC_STAT_LEFT], l["stats"][cv.CC_STAT_TOP])
                pt2 = (pt1[0] + l["stats"][cv.CC_STAT_WIDTH], pt1[1] + l["stats"][cv.CC_STAT_HEIGHT])
                cv.rectangle(img_output, pt1, pt2, (255, 0, 0), 3)

            pixmap = self.__opencv2pixmap(img_output)

        return pixmap

    def __setup_ref_algorithm(self):
        self.__alg = ReferenceAlgorithm(self.__img_path)
        
        self.__alg.s_thresh_level = self.__window.sThresholdSpin.value()
        self.__alg.v_thresh_level = self.__window.vThresholdSpin.value()
        self.__alg.min_blob_size = self.__window.refMinSizeSpin.value()

    def __setup_custom_algorithm(self):
        self.__alg = CustomHsvBlobAlgorithm(self.__img_path)

        self.__alg.min_blob_size = self.__window.customMinSizeSpin.value()
        self.__alg.safe_area = 1 - self.__window.safeAreaSpin.value() / 100

    def __opencv2pixmap(self, img, format=QImage.Format_BGR888) -> QPixmap:
        if len(img.shape) == 3:
            h, w, c = img.shape
        else:
            h, w = img.shape
            c = 1
        d = img.data

        return QPixmap(QImage(d, w, h, c * w, format))
