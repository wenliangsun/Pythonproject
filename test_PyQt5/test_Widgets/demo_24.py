"""进度条示例"""

import sys
from PyQt5 import QtWidgets, QtGui, QtCore


class ProgressBar(QtWidgets.QWidget):
    def __init__(self):
        super(ProgressBar, self).__init__()

        self.setWindowTitle("进度条演示程序")
        self.setGeometry(300, 300, 250, 150)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setGeometry(30, 40, 200, 25)

        self.button = QtWidgets.QPushButton("开始", self)
        self.button.setFocusPolicy(QtCore.Qt.NoFocus)
        self.button.move(40, 80)
        self.button.clicked.connect(self.on_start)

        self.timer = QtCore.QBasicTimer() #创建一个定时器对象。该方法的第一个参数为超时时间。第二个参数表示当前超时时间到了以后定时器触发超时事件的接收对象。
        self.step = 0

    def timerEvent(self, *args, **kwargs):
        if self.step >= 100:
            self.timer.stop()
            return
        self.step += 1
        self.progress_bar.setValue(self.step)

    def on_start(self):
        if self.timer.isActive():
            self.timer.stop()
            self.button.setText("开始")
        else:
            self.timer.start(100, self)
            self.button.setText("停止")


app = QtWidgets.QApplication(sys.argv)
pb = ProgressBar()
pb.show()
sys.exit(app.exec_())
