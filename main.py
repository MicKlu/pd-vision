#!/usr/bin/python3

import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.Qt import QMimeDatabase
from ui.main_window import Ui_MainWindow
from worker import CountingWorker, CountingWorkerError

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.__worker = None

        self.setupUi(self)
        self.__connect_signals_and_slots()

    def __connect_signals_and_slots(self):
        self.safeAreaSlider.valueChanged.connect(self.__safe_area_slider2spin)
        self.safeAreaSpin.valueChanged.connect(self.__safe_area_spin2slider)

        self.actionExit.triggered.connect(self.__quit)
        self.actionOpen.triggered.connect(self.__open)

        self.countButton.clicked.connect(self.__count)

    def __open(self):
        mimes = ["image/jpeg", "image/png"]
        
        mime_db = QMimeDatabase()
        filter = []
        for m in mimes:
            filter += mime_db.mimeTypeForName(m).globPatterns()
        filter = f"Wspierane obrazy ({' '.join(filter)})"

        file_path = QFileDialog.getOpenFileName(self, filter=filter)[0]

        try:
            self.__worker = CountingWorker(self, file_path)

            self.countButton.setEnabled(True)
            self.algorithmCombo.setEnabled(True)
            self.imageComboLeft.setCurrentIndex(0)
            self.imageComboLeft.setEnabled(False)
            self.imageComboRight.setCurrentIndex(self.imageComboRight.count() - 1)
            self.imageComboRight.setEnabled(False)
        except CountingWorkerError as e:
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("Błąd")
            msgBox.setText(e.args[0])
            msgBox.setStandardButtons(QMessageBox.Ok)
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.exec()

    def __count(self):
        self.__worker.count()

    def __quit(self):
        self.close()

    def __safe_area_slider2spin(self, value: int):
        self.safeAreaSpin.setValue(value / 100)

    def __safe_area_spin2slider(self, value: int):
        self.safeAreaSlider.setValue(round(value * 100))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())

