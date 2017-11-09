# -*- coding: utf-8 -*-

"""
Module implementing MainWindow.
"""
from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import time

from Ui_test_pyqt import Ui_MainWindow


class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Class documentation goes here.
    """

    def __init__(self, parent=None):
        """
        Constructor
        
        @param parent reference to the parent widget
        @type QWidget
        """
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        time.sleep(2)

    @pyqtSlot()
    def on_pushButton_6_clicked(self):
        """
        Slot documentation goes here.
        """
        # TODO: not implemented yet
        print("Press")

    @pyqtSlot()
    def on_test_clicked(self):
        """
        Slot documentation goes here.
        """
        print(self.textBrowser.toPlainText())


if __name__ == "__main__":
    import sys
    from PyQt5 import QtWidgets

    app = QtWidgets.QApplication(sys.argv)
    splash = QSplashScreen(QPixmap(":/my_pic/logo.png"))
    splash.show()
    splash.showMessage("正在加载图片资源...", QtCore.Qt.AlignBottom)
    time.sleep(2)
    splash.showMessage("正在加载音频资源...", QtCore.Qt.AlignBottom)
    time.sleep(2)
    splash.showMessage("正在渲染界面...", QtCore.Qt.AlignBottom)
    time.sleep(2)
    app.processEvents()

    ui = MainWindow()
    ui.show()
    splash.finish(ui)

    sys.exit(app.exec_())
