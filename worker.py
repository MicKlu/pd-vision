import cv2 as cv

from PyQt5.Qt import Qt
from PyQt5.Qt import QImage, QPixmap

class CountingWorkerError(Exception):

    def __init__(self, args):
        super().__init__(args)

class CountingWorker:

    def __init__(self, window: 'MainWindow', img_path: str):
        self.__window = window
        self.__img_path = img_path

        img = cv.imread(img_path)
        if img is None:
            raise CountingWorkerError("Plik nie może być otwarty")

        h, w, c = img.shape
        d = img.data

        
        pixmap = QPixmap(QImage(d, w, h, c * w, QImage.Format_BGR888))
        window.imagePreviewLeft.setImage(pixmap)
        
        

        

        