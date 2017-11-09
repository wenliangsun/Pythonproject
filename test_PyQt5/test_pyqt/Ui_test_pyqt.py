# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '/home/sunwl/Projects/Eric6project/test/test_pyqt.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(708, 627)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.pushButton = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton.setGeometry(QtCore.QRect(620, 500, 80, 22))
        self.pushButton.setObjectName("pushButton")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_2.setGeometry(QtCore.QRect(620, 530, 80, 22))
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_3.setGeometry(QtCore.QRect(620, 560, 80, 22))
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_4.setGeometry(QtCore.QRect(520, 500, 80, 22))
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_5.setGeometry(QtCore.QRect(520, 530, 80, 22))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_6.setGeometry(QtCore.QRect(520, 560, 80, 22))
        self.pushButton_6.setObjectName("pushButton_6")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralWidget)
        self.textBrowser.setGeometry(QtCore.QRect(50, 40, 221, 211))
        self.textBrowser.setReadOnly(False)
        self.textBrowser.setObjectName("textBrowser")
        self.pushButton_7 = QtWidgets.QPushButton(self.centralWidget)
        self.pushButton_7.setGeometry(QtCore.QRect(90, 490, 80, 22))
        self.pushButton_7.setObjectName("test")
        self.graphicsView = QtWidgets.QGraphicsView(self.centralWidget)
        self.graphicsView.setGeometry(QtCore.QRect(480, 30, 171, 161))
        self.graphicsView.setStyleSheet("border-image: url(:/my_pic/640-3.jpg);")
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(self.centralWidget)
        self.label.setGeometry(QtCore.QRect(420, 260, 221, 151))
        self.label.setText("")
        self.label.setObjectName("label")
        MainWindow.setCentralWidget(self.centralWidget)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked.connect(MainWindow.close)
        self.pushButton_2.clicked.connect(MainWindow.showFullScreen)
        self.pushButton_3.clicked.connect(MainWindow.showNormal)
        self.pushButton_4.clicked.connect(MainWindow.showMinimized)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "close"))
        self.pushButton_2.setText(_translate("MainWindow", "fullscreen"))
        self.pushButton_3.setText(_translate("MainWindow", "recover"))
        self.pushButton_4.setText(_translate("MainWindow", "min"))
        self.pushButton_5.setText(_translate("MainWindow", "change"))
        self.pushButton_6.setText(_translate("MainWindow", "test"))
        self.pushButton_7.setText(_translate("MainWindow", "print"))


import my_test_rc

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
