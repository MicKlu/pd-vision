import os
import tempfile
from pathlib import Path
import shutil

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from PyQt5.Qt import Qt
from PyQt5.Qt import QImage, QPixmap

from alg.ref import ReferenceAlgorithm, ReferenceAlgorithmToleranceCount
from alg.custom import CustomHsvBlobAlgorithm

class CountingWorkerError(Exception):

    def __init__(self, args):
        super().__init__(args)

class CountingWorker:

    ALGORITHM_REF = 0
    ALGORITHM_CUSTOM = 1

    IMAGE_ORIGINAL = 0
    IMAGE_ORIGINAL_H = 1
    IMAGE_ORIGINAL_S = 2
    IMAGE_ORIGINAL_V = 3
    IMAGE_PREPROCESSED = 4
    IMAGE_PREPROCESSED_H = 5
    IMAGE_PREPROCESSED_S = 6
    IMAGE_PREPROCESSED_V = 7
    IMAGE_THRESHOLD_H = 8
    IMAGE_THRESHOLD_S = 9
    IMAGE_THRESHOLD_V = 10
    IMAGE_MORPHED = 11
    IMAGE_COUNTING = 12

    def __init__(self, window: 'MainWindow', img_path: str):
        self.__window = window
        self.__img_path = img_path
        self.__alg = None
        self.__alg_id = None

        self.__img = cv.imread(img_path)
        if self.__img is None:
            raise CountingWorkerError("Plik nie może być otwarty")

        pixmap = self.__opencv2pixmap(self.__img)
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

    def save_image(self, save_path: str, image_index: int):
        img = self.get_image(image_index)

        save_path = Path(save_path)
        temp_file = str(Path.joinpath(Path(tempfile.gettempdir()), (save_path.stem + ".png")))
        cv.imwrite(temp_file, img)

        try:
            shutil.move(temp_file, save_path)
        except PermissionError:
            os.remove(temp_file)
            raise CountingWorkerError("Nie można zapisać pliku: brak dostępu")
        except:
            raise CountingWorkerError("Nie można zapisać pliku")

    def show_image(self, image_index: int):
        name = "preview"
        cv.namedWindow(name, cv.WINDOW_NORMAL)
        cv.setWindowTitle(name, "Podgląd obrazu")
        cv.imshow(name, self.get_image(image_index))

    def show_histogram(self, image_index: int):
        img = self.get_image(image_index)

        if img is None:
            return

        channels = 1
        bins = 256

        if (image_index in [
                CountingWorker.IMAGE_ORIGINAL,
                CountingWorker.IMAGE_PREPROCESSED,
                CountingWorker.IMAGE_COUNTING
                ]):
            channels = 3

        if (image_index in [
                CountingWorker.IMAGE_ORIGINAL_H,
                CountingWorker.IMAGE_PREPROCESSED_H,
                CountingWorker.IMAGE_THRESHOLD_H,
                ]):
            bins = 180

        plt.figure("Histogram", figsize=(4,2), dpi=72, clear=True)
        plt.ion()

        if channels == 1:
            plt.hist(img.ravel(), bins, [0, bins])
        elif channels == 3:
            hist_r = cv.calcHist([img],[2],None,[256],[0,256])
            hist_g = cv.calcHist([img],[1],None,[256],[0,256])
            hist_b = cv.calcHist([img],[0],None,[256],[0,256])
            plt.plot(hist_r, "r")
            plt.plot(hist_g, "g")
            plt.plot(hist_b, "b")

        # max = np.max(np.histogram(img.ravel(), bins, [0,bins])[0])
        # if image_index == CountingWorker.IMAGE_PREPROCESSED_S:
        #     plt.bar([self.__alg.s_thresh_level], [max], color="r")

        # if image_index == CountingWorker.IMAGE_PREPROCESSED_V:
        #     plt.bar([self.__alg.v_thresh_level], [max], color="r")


        plt.xlabel(r"Poziom jasności $r_k$")
        plt.ylabel(r"Liczba pikseli $h(r_k)$")

        plt.xlim(0, bins - 1)

        plt.tight_layout(pad=0.01)

        plt.show()

    def get_image(self, image_index: int):
        img = None

        if image_index == CountingWorker.IMAGE_ORIGINAL:
            img = self.__img

        if self.__alg is None:
            return img

        if (image_index >= CountingWorker.IMAGE_ORIGINAL_H
                and image_index <= CountingWorker.IMAGE_ORIGINAL_V):

            img_hsv = cv.cvtColor(self.__alg.img_original_bgr, cv.COLOR_BGR2HSV)
            (h,s,v) = cv.split(img_hsv)

            if image_index == CountingWorker.IMAGE_ORIGINAL_H:
                img = h
            elif image_index == CountingWorker.IMAGE_ORIGINAL_S:
                img = s
            elif image_index == CountingWorker.IMAGE_ORIGINAL_V:
                img = v
        else:
            if image_index == CountingWorker.IMAGE_PREPROCESSED:
                img = self.__alg.img_prep_bgr
            elif image_index == CountingWorker.IMAGE_PREPROCESSED_H:
                img = self.__alg.img_prep_h
            elif image_index == CountingWorker.IMAGE_PREPROCESSED_S:
                img = self.__alg.img_prep_s
            elif image_index == CountingWorker.IMAGE_PREPROCESSED_V:
                img = self.__alg.img_prep_v
            elif image_index == CountingWorker.IMAGE_THRESHOLD_H:
                img = self.__alg.img_h_thresh
            elif image_index == CountingWorker.IMAGE_THRESHOLD_S:
                img = self.__alg.img_s_thresh
            elif image_index == CountingWorker.IMAGE_THRESHOLD_V:
                img = self.__alg.img_v_thresh
            elif image_index == CountingWorker.IMAGE_MORPHED:
                img = self.__alg.img_morphed
            elif image_index == CountingWorker.IMAGE_COUNTING:
                img_output = np.copy(self.__alg.img_original_bgr)
                # cv.drawContours(img_output, self.__alg.blobs, -1, (0, 0, 255), 3)

                for l in self.__alg.valid_blobs:
                    pt1 = (l["stats"][cv.CC_STAT_LEFT], l["stats"][cv.CC_STAT_TOP])
                    pt2 = (pt1[0] + l["stats"][cv.CC_STAT_WIDTH], pt1[1] + l["stats"][cv.CC_STAT_HEIGHT])
                    cv.rectangle(img_output, pt1, pt2, (255, 0, 0), 2)

                img = img_output

        return img

    def get_pixmap(self, image_index: int):
        pixmap = QPixmap()

        if self.__alg is None:
            return pixmap

        img = self.get_image(image_index)

        if ((image_index >= CountingWorker.IMAGE_ORIGINAL_H
                and image_index <= CountingWorker.IMAGE_ORIGINAL_V)
                or (image_index >= CountingWorker.IMAGE_PREPROCESSED_H
                and image_index <= CountingWorker.IMAGE_MORPHED)):
            pixmap = self.__opencv2pixmap(img, QImage.Format_Grayscale8)
        else:
            pixmap = self.__opencv2pixmap(img)

        return pixmap

    def __setup_ref_algorithm(self):
        self.__alg = ReferenceAlgorithm(self.__img_path)
        # self.__alg = ReferenceAlgorithmToleranceCount(self.__img_path)

        
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
