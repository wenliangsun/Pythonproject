"""拖放功能示例"""

import sys

from PyQt5 import QtWidgets


class Button(QtWidgets.QPushButton):
    def __init__(self, title, parent):
        super(Button, self).__init__(title, parent)
        self.setAcceptDrops(True)  # 允许QPushButton的放置事件（即将其它对象拖放到其上的操作）。

    def dragEnterEvent(self, event):
        """
        重写dragEnterEvent()方法，我们通知我们将要收到的数据类型，这里是无格式文本
        """
        if event.mimeData().hasFormat("text/plain"):
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        """
        重写dropEvent()方法，我们定义我们收到drop事件后如何操作，这里我们改变按钮组件显示的文本。
        """
        self.setText(event.mimeData().text())


class DragDrop(QtWidgets.QDialog):
    def __init__(self):
        super(DragDrop, self).__init__()

        self.setWindowTitle("拖放功能演示程序")
        self.resize(280, 150)
        edit = QtWidgets.QLineEdit("", self)
        edit.move(30, 65)
        edit.setDragEnabled(True)  # QLineEdit组件有内置的拖动操作，我们所要做的就是调用setDragEnabled()方法来将其设置为可用。
        button = Button("按钮", self)
        button.move(170, 65)

        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)


app = QtWidgets.QApplication(sys.argv)
dd = DragDrop()
dd.show()
sys.exit(app.exec_())
