"""将窗口放在屏幕中心"""

import sys
from PyQt5 import QtWidgets


class Center(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Center, self).__init__(parent)
        self.setWindowTitle("窗口置中程序")
        self.resize(250, 150)
        self.center()

    def center(self):
        # 该语句用来计算出显示器的分辨率（screen.width, screen.height）。
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2,
                  (screen.height() - size.height()) // 2)


app = QtWidgets.QApplication(sys.argv)
center = Center()
center.show()
sys.exit(app.exec_())
