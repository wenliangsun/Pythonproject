"""绘图中的颜色示例"""
import sys
from PyQt5 import QtWidgets, QtCore, QtGui


class Color(QtWidgets.QWidget):
    def __init__(self):
        super(Color, self).__init__()

        self.setWindowTitle("绘图中的颜色示例程序")
        self.setGeometry(300, 300, 350, 300)

    def paintEvent(self, event: QtGui.QPaintEvent):
        paint = QtGui.QPainter()
        paint.begin(self)

        paint.setBrush(QtGui.QColor(255, 0, 0, 0))
        paint.drawRect(10, 15, 90, 60)

        paint.setBrush(QtGui.QColor(255, 0, 0, 160))
        paint.drawRect(130, 15, 90, 60)

        paint.setBrush(QtGui.QColor(255, 0, 0, 255))
        paint.drawRect(250, 15, 90, 60)

        paint.setBrush(QtGui.QColor(0, 255, 2, 80))
        paint.drawRect(10, 105, 90, 60)

        paint.setBrush(QtGui.QColor(0, 255, 0, 160))
        paint.drawRect(130, 105, 90, 60)

        paint.setBrush(QtGui.QColor(0, 255, 0, 255))
        paint.drawRect(250, 105, 90, 60)

        paint.setBrush(QtGui.QColor(0, 0, 255, 60))
        paint.drawRect(10, 195, 90, 60)

        paint.setBrush(QtGui.QColor(0, 0, 255, 160))
        paint.drawRect(130, 195, 90, 60)

        paint.setBrush(QtGui.QColor(0, 0, 255, 255))
        paint.drawRect(250, 195, 90, 60)

        paint.end()


app = QtWidgets.QApplication(sys.argv)
c = Color()
c.show()
sys.exit(app.exec_())
