import sys

from PyQt5 import QtWidgets,QtGui
from demo_09 import Ui_Form
from PyQt5.QtWidgets import QFileDialog

class myWindow(QtWidgets.QWidget,Ui_Form):
    def __init__(self):
        super(myWindow, self).__init__()
        self.setupUi(self)

    def openimage(self):
        # 打开文件路径
        # 设置文件扩展名过滤,注意用双分号间隔
        imgName, imgType = QFileDialog.getOpenFileName(self,
                                                       "打开图片",
                                                       "",
                                                       " *.jpg;;*.png;;*.jpeg;;*.bmp;;All Files (*)")

        print(imgName)
        # 利用qlabel显示图片
        png = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(png)

app = QtWidgets.QApplication(sys.argv)
window = myWindow()
window.show()
sys.exit(app.exec_())