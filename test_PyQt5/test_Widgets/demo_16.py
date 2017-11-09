"""发射信号示例"""

import sys

from PyQt5 import QtWidgets, QtCore


class EmitSignal(QtWidgets.QWidget):
    # 创建一个叫做closeEmitApp()的新的信号。这个信号在鼠标按下时产生。
    closeEmitApp = QtCore.pyqtSignal()

    def __init__(self):
        super(EmitSignal, self).__init__()

        self.setWindowTitle("发射信号演示程序")
        self.resize(250, 150)

        self.closeEmitApp.connect(self.close)

    def mousePressEvent(self, QMouseEvent):
        self.closeEmitApp.emit()


app = QtWidgets.QApplication(sys.argv)
es = EmitSignal()
es.show()
sys.exit(app.exec_())
