import cv2 as cv

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

    def __init__(self, window: 'MainWindow', img_path: str):
        self.__window = window
        self.__img_path = img_path

        img = cv.imread(img_path)
        if img is None:
            raise CountingWorkerError("Plik nie może być otwarty")

        pixmap = self.__opencv2pixmap(img)
        window.imagePreviewLeft.setImage(pixmap)

    def count(self):
        alg_id = self.__window.algorithmCombo.currentIndex()
        
        if alg_id == CountingWorker.ALGORITHM_REF:
            self.__setup_ref_algorithm()
        elif alg_id == CountingWorker.ALGORITHM_CUSTOM:
            self.__setup_custom_algorithm()

        e1 = cv.getTickCount()
        
        count = self.__alg.count()
        
        e2 = cv.getTickCount()
        time = (e2 - e1)/ cv.getTickFrequency()

        self.__window.detectedValueLabel.setText(f"{count}")
        self.__window.executionTimeValueLabel.setText(f"{time} s")


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
        h, w, c = img.shape
        d = img.data

        return QPixmap(QImage(d, w, h, c * w, format))
