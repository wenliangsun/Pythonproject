"""PyQt5中的菜单和工具栏"""
# QMainWindow 类用来创建应用程序的主窗口。
# 通过该类，我们可以创建一个包含状态栏、工具栏和菜单栏的经典应用程序框架。

# 状态栏
# 状态栏是用来显示状态信息的串口部件。

import sys
from PyQt5 import QtWidgets, QtGui


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(400, 300)
        self.setWindowTitle("状态栏程序示例")
        self.statusBar().showMessage("就绪")


app = QtWidgets.QApplication(sys.argv)
main_window = MainWindow()
main_window.show()
sys.exit(app.exec_())
