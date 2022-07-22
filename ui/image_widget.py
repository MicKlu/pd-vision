from PyQt5.Qt import Qt, QPixmap
from PyQt5.QtWidgets import QLabel

class ImageWidget(QLabel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.__pixmap = QPixmap()

    def resizeEvent(self, event):
        self.__scalePixmap()

    def setImage(self, pixmap: QPixmap):
        self.__pixmap = pixmap
        
        if self.__pixmap.isNull():
            self.setPixmap(pixmap)
            return

        self.__scalePixmap()

    def __scalePixmap(self):
        if self.__pixmap.isNull():
            return

        scaled_pixmap = self.__pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        self.setPixmap(scaled_pixmap)
