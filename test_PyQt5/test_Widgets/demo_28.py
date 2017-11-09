"""绘制文本示例"""

import sys

from PyQt5 import QtWidgets, QtGui, QtCore


class DrawText(QtWidgets.QWidget):
    def __init__(self):
        super(DrawText, self).__init__()

        self.setWindowTitle("绘制文本演示程序")
        self.setGeometry(300, 300, 250, 150)

        self.text = "大家好，\n西安电子科技大学。"

    def paintEvent(self, event: QtGui.QPaintEvent):
        # 通过一个绘图事件实现绘图。
        paint = QtGui.QPainter()
        paint.begin(self)
        paint.setPen(QtGui.QColor(168, 34, 3))
        paint.setFont(QtGui.QFont("STLiti", 20))
        paint.drawText(event.rect(), QtCore.Qt.AlignCenter, self.text)
        paint.end()


app = QtWidgets.QApplication(sys.argv)
dt = DrawText()
dt.show()
sys.exit(app.exec_())
