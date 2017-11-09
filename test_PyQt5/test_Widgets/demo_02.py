"""设置程序图标"""
import sys
from PyQt5 import QtWidgets, QtGui


class Icon(QtWidgets.QWidget):  # Icon --> 图标
    def __init__(self, parent=None):
        super(Icon, self).__init__(parent)
        # QtWidgets.QWidget.__init__(self, parent)

        self.setGeometry(300, 300, 250, 150)  # 设置窗口的位置
        self.setWindowTitle("icon")
        # 设置程序图标它需要一个QIcon类型的对象作为参数。
        # 调用QIcon构造函数时，我们需要提供要显示的图标的路径（相对或绝对路径）。
        self.setWindowIcon(QtGui.QIcon(r"../images/new_on.jpg"))


app = QtWidgets.QApplication(sys.argv)
icon = Icon()
icon.show()
sys.exit(app.exec_())
