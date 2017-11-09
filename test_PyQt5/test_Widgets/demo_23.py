"""滑块和标签示例"""

import sys

from PyQt5 import QtWidgets, QtCore, QtGui


class SliderLabel(QtWidgets.QWidget):
    def __init__(self):
        super(SliderLabel, self).__init__()

        self.setWindowTitle("滑块和标签演示过程")
        self.setGeometry(300, 300, 300, 200)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)  # 创建了一个水平滑块部件。
        self.slider.setGeometry(30, 40, 100, 30)
        self.slider.setFocusPolicy(QtCore.Qt.NoFocus)
        self.slider.valueChanged.connect(self.change_value)

        self.label = QtWidgets.QLabel(self)
        self.label.setPixmap(QtGui.QPixmap(r"../images/new_on.jpg"))  # 创建一个标签部件并将 sample_06.png 加入到该部件中显示。
        self.label.setGeometry(160, 40, 128, 128)

    def change_value(self):
        pos = self.slider.value()
        if pos == 0:
            self.label.setPixmap(QtGui.QPixmap(r"../images/new_on.jpg"))
        elif 0 < pos < 60:
            self.label.setPixmap(QtGui.QPixmap(r"../images/new_login.jpg"))
        else:
            self.label.setPixmap(QtGui.QPixmap(r"../images/new_logout.jpg"))


app = QtWidgets.QApplication(sys.argv)
sl = SliderLabel()
sl.show()
sys.exit(app.exec_())
