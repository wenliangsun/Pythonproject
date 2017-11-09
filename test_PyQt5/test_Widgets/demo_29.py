"""绘制散点示例"""

import sys, math

from PyQt5 import QtWidgets, QtGui, QtCore


class DrawPoints(QtWidgets.QWidget):
    def __init__(self):
        super(DrawPoints, self).__init__()

        self.setWindowTitle("绘制散点演示程序")
        self.setGeometry(300, 300, 600, 300)

    def paintEvent(self, event: QtGui.QPaintEvent):
        paint = QtGui.QPainter()
        paint.begin(self)
        paint.setPen(QtCore.Qt.red)
        size = self.size()
        for i in range(0, size.width(), 3):
            x = i
            y = int(math.sqrt(300 ** 2 - (x - 300) ** 2))
            paint.drawPoint(x, y)
        paint.end()


app = QtWidgets.QApplication(sys.argv)
dp = DrawPoints()
dp.show()
sys.exit(app.exec_())
