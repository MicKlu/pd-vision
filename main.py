#!/usr/bin/python3

import sys
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.Qt import QMimeDatabase
from ui.main_window import Ui_MainWindow

from matplotlib import pyplot as plt

from worker import CountingWorker, CountingWorkerError

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.__worker = None
        self.report_file = None

        self.setupUi(self)
        self.__connect_signals_and_slots()

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)

        self.imagePreviewLeft.addAction(self.actionShowImageLeft)
        self.imagePreviewRight.addAction(self.actionShowImageRight)

        self.imagePreviewLeft.addAction(self.actionShowHistogramLeft)
        self.imagePreviewRight.addAction(self.actionShowHistogramRight)

        self.imagePreviewLeft.addAction(self.actionSaveLeft)
        self.imagePreviewRight.addAction(self.actionSaveRight)

        self.actionSaveLeft.setEnabled(False)
        self.actionSaveRight.setEnabled(False)
        self.actionShowHistogramLeft.setEnabled(False)
        self.actionShowHistogramRight.setEnabled(False)
        self.actionShowImageLeft.setEnabled(False)
        self.actionShowImageRight.setEnabled(False)

    def __connect_signals_and_slots(self):
        self.safeAreaSlider.valueChanged.connect(self.__safe_area_slider2spin)
        self.safeAreaSpin.valueChanged.connect(self.__safe_area_spin2slider)

        self.actionExit.triggered.connect(self.__quit)
        self.actionOpen.triggered.connect(self.__open)

        self.countButton.clicked.connect(self.__count)

        self.imageComboLeft.currentIndexChanged.connect(self.update_image_left)
        self.imageComboRight.currentIndexChanged.connect(self.update_image_right)

        self.actionSaveLeft.triggered.connect(self.__save_image_left)
        self.actionSaveRight.triggered.connect(self.__save_image_right)

        self.actionShowHistogramLeft.triggered.connect(self.__show_histogram_left)
        self.actionShowHistogramRight.triggered.connect(self.__show_histogram_right)

        self.actionShowImageLeft.triggered.connect(self.__show_image_left)
        self.actionShowImageRight.triggered.connect(self.__show_image_right)

    def __open(self):
        mimes = ["image/jpeg", "image/png"]
        
        mime_db = QMimeDatabase()
        filter = []
        for m in mimes:
            filter += mime_db.mimeTypeForName(m).globPatterns()
        filter = f"Wspierane obrazy ({' '.join(filter)})"

        file_path = QFileDialog.getOpenFileName(self, filter=filter)[0]

        if file_path == "":
            return

        try:
            self.__worker = CountingWorker(self, file_path)

            self.countButton.setEnabled(True)
            self.algorithmCombo.setEnabled(True)
            self.imageComboLeft.setCurrentIndex(0)
            self.imageComboLeft.setEnabled(False)
            self.imageComboRight.setCurrentIndex(self.imageComboRight.count() - 1)
            self.imageComboRight.setEnabled(False)
            self.update_image_right(self.imageComboRight.currentIndex())

            self.actionSaveLeft.setEnabled(True)
            self.actionSaveRight.setEnabled(False)
            self.actionShowHistogramLeft.setEnabled(True)
            self.actionShowHistogramRight.setEnabled(False)
            self.actionShowImageLeft.setEnabled(True)
            self.actionShowImageRight.setEnabled(False)

        except CountingWorkerError as e:
            self.__show_warning("Błąd", e.args[0])

    def __count(self):
        self.__worker.count()

        self.imageComboLeft.setEnabled(True)
        self.imageComboRight.setEnabled(True)

        self.actionSaveRight.setEnabled(True)
        self.actionShowHistogramRight.setEnabled(True)
        self.actionShowImageRight.setEnabled(True)

    def __quit(self):
        self.close()

    def __get_image_index(self, preview_id: int) -> int:
        if preview_id == 0:
            return self.imageComboLeft.currentIndex()
        elif preview_id == 1:
            return self.imageComboRight.currentIndex()

    def __save_image(self, preview_id: int):
        image_index = self.__get_image_index(preview_id)

        filter = f"Obraz PNG (*.png)"

        save_path = QFileDialog.getSaveFileName(filter=filter)[0]

        if save_path == "":
            return

        try:
            self.__worker.save_image(save_path, image_index)
        except CountingWorkerError as e:
            self.__show_warning("Błąd", e.args[0])

    def __save_image_left(self):
        self.__save_image(0)

    def __save_image_right(self):
        self.__save_image(1)

    def __show_image(self, preview_id: int):
        image_index = self.__get_image_index(preview_id)

        self.__worker.show_image(image_index)

    def __show_image_left(self):
        self.__show_image(0)

    def __show_image_right(self):
        self.__show_image(1)

    def __show_histogram(self, preview_id: int):
        image_index = self.__get_image_index(preview_id)

        self.__worker.show_histogram(image_index)

    def __show_histogram_left(self):
        self.__show_histogram(0)

    def __show_histogram_right(self):
        self.__show_histogram(1)

    def __update_image_preview(self, preview_id: int, image_index: int):
        pixmap = self.__worker.get_pixmap(image_index)

        if preview_id == 0:
            self.imagePreviewLeft.setImage(pixmap)
        elif preview_id == 1:
            self.imagePreviewRight.setImage(pixmap)

    def update_image_left(self, image_index=None):
        if image_index is None:
            image_index = self.imageComboLeft.currentIndex()
        self.__update_image_preview(0, image_index)

    def update_image_right(self, image_index=None):
        if image_index is None:
            image_index = self.imageComboRight.currentIndex()
        self.__update_image_preview(1, image_index)

    def __safe_area_slider2spin(self, value: int):
        self.safeAreaSpin.setValue(value / 100)

    def __safe_area_spin2slider(self, value: int):
        self.safeAreaSlider.setValue(round(value * 100))

    def __show_warning(self, title: str, text: str):
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle(title)
        msgBox.setText(text)
        msgBox.setStandardButtons(QMessageBox.Ok)
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.exec()

    def closeEvent(self, ev):
        super().closeEvent(ev)
        plt.close('all')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

