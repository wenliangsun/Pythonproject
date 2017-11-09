"""
消息窗口
默认情况下，如果我们单击了窗口标题栏上的X标记，窗口就会被关闭。
但是有些时候我们想要改变这一默认行为。
比如，我们正在编辑的文件内容发生了变化，这时若单击X标记关闭窗口，
编辑器就应当弹出确认窗口。
"""

import sys
from PyQt5 import QtWidgets, QtCore, QtGui


class MessageBox(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(MessageBox, self).__init__(parent)
        self.setGeometry(200, 200, 250, 150)
        self.setWindowTitle("The Message Window")


    def closeEvent(self, event): # 钩子函数  回调函数的感觉
        """
        如果我们关闭QWidget窗口，QCloseEvent事件就会被触发。
        要改变原有的wdiget行为阻止查窗口的关闭，我们就需要重新实现closeEvent()方法。
        """
        reply = QtWidgets.QMessageBox.question(self, "确认退出", "你确定要退出么？",
                                               QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

app = QtWidgets.QApplication(sys.argv)
messageBox = MessageBox()
messageBox.show()
sys.exit(app.exec_())
