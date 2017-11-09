"""
关闭窗口
一个显而易见的关闭窗口的方式是单击标题栏右上角的X标记。
在接下来的示例中，我们将展示如何用代码来关闭程序，并简要介绍Qt的信号和槽机制。
"""

import sys
from PyQt5 import QtWidgets, QtGui, QtCore


class QuitButton(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(QuitButton, self).__init__(parent=parent)

        self.setGeometry(200, 200, 250, 150)
        self.setWindowTitle("My Quit Program")

        # 用来创建一个按钮并将其放在QWidget部件上，
        # 就像我们将QWidget部件放在屏幕上一样。
        quit_button = QtWidgets.QPushButton("Quit", self)
        quit_button.setGeometry(10, 10, 60, 35)

        quit_button.clicked.connect(QtWidgets.qApp.quit)


app = QtWidgets.QApplication(sys.argv)
quitbutton = QuitButton()
quitbutton.show()
sys.exit(app.exec_())
