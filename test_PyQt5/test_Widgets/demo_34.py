"""背景图自适应窗体大小的类"""

import sys

from PyQt5 import QtWidgets, QtGui, QtCore


class CentralWidget(QtWidgets.QWidget):
    def __init__(self):
        super(CentralWidget, self).__init__()

        # 设置背景图片
        self.background = QtGui.QImage(r"../images/new_on.jpg")
        self.setAutoFillBackground(True)
        # self.setStyleSheet("QWidget{background-image:../images/new_on.jpg}")

    def resizeEvent(self, event: QtGui.QResizeEvent):
        # 重写resizeEvent, 使背景图案可以根据窗口大小改变
        QtWidgets.QWidget.resizeEvent(self, event)
        palette = QtGui.QPalette()
        palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(self.background.scaled(event.size())))
        self.setPalette(palette)


app = QtWidgets.QApplication(sys.argv)
cw = CentralWidget()
cw.show()
sys.exit(app.exec_())
