"""Qpen示例"""

import sys

from PyQt5 import QtWidgets, QtCore, QtGui


class Brush(QtWidgets.QWidget):
    def __init__(self):
        super(Brush, self).__init__()

        self.setWindowTitle("Qpen示例演示程序")
        self.setGeometry(300, 300, 280, 270)

    def paintEvent(self, event: QtGui.QPaintEvent):
        paint = QtGui.QPainter()
        paint.begin(self)

        pen = QtGui.QPen(QtCore.Qt.black, 2, QtCore.Qt.SolidLine)

        paint.setPen(pen)
        paint.drawLine(20, 40, 250, 40)

        pen.setStyle(QtCore.Qt.DashLine)
        paint.setPen(pen)
        paint.drawLine(20, 80, 250, 80)

        pen.setStyle(QtCore.Qt.DashDotLine)
        paint.setPen(pen)
        paint.drawLine(20, 120, 250, 120)

        pen.setStyle(QtCore.Qt.DotLine)
        paint.setPen(pen)
        paint.drawLine(20, 160, 250, 160)

        pen.setStyle(QtCore.Qt.DashDotDotLine)
        paint.setPen(pen)
        paint.drawLine(20, 200, 250, 200)

        pen.setStyle(QtCore.Qt.CustomDashLine)  # 自定义
        # 数字列表定义一个类型，必须有偶数个数字，奇数位置定义一个破折号，
        # 偶数位置定义一个空白，数字越大，空白或者破折号就越长。
        pen.setDashPattern([1, 4, 5, 4])
        paint.setPen(pen)
        paint.drawLine(20, 240, 250, 240)

        paint.end()


app = QtWidgets.QApplication(sys.argv)
ps = Brush()
ps.show()
sys.exit(app.exec_())
