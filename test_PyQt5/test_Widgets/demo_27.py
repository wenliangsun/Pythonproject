"""拖放按钮部件示例"""

import sys

from PyQt5 import QtWidgets, QtGui, QtCore


class Button(QtWidgets.QPushButton):
    def __init__(self, title, parent):
        super(Button, self).__init__(title, parent)

    def mouseMoveEvent(self, event):
        if event.buttons() != QtCore.Qt.LeftButton:
            return
        """
        这段代码其实是定义了Button类的拖放属性，
        首先定义一个QMimeData类的示例用于存存储在拖放过程中的信息，
        可以将需要的信息通过QMimeData传到被拖放到的部件处；
        然后定义了一个QDrag类，并用setMimeData()方法设定其将拖放信息存放在mime_data中。
        """
        mima_data = QtCore.QMimeData()  # 创建一个QDrag对象
        drag = QtGui.QDrag(self)
        drag.setMimeData(mima_data)
        # print(self.rect().topLeft().x(),self.rect().topLeft().y())
        drag.setHotSpot(event.pos() - self.rect().topLeft())
        # 拖动对象的exec_ ()方法开始拖放操作。如果我们完成一次移动放置操作，
        # 我们要销毁按钮组件。技术上来讲，我们在当前位置销毁一个组件，
        # 并在新位置重新创建它。
        drop_action = drag.exec_(QtCore.Qt.MoveAction)

        if drop_action == QtCore.Qt.MoveAction:
            self.close()

    def mousePressEvent(self, event):
        QtWidgets.QPushButton.mousePressEvent(self, event)
        if event.button() == QtCore.Qt.LeftButton:
            print("按下")


class DragButton(QtWidgets.QDialog):
    def __init__(self):
        super(DragButton, self).__init__()
        self.setWindowTitle("拖放按钮演示程序")
        self.resize(280, 150)
        self.setAcceptDrops(True)
        self.button = Button("关闭", self)
        self.button.move(100, 65)

        screen = QtWidgets.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        position = event.pos()
        button = Button("关闭", self)
        button.move(position)
        button.show()
        event.setDropAction(QtCore.Qt.MoveAction)  # 指定释放操作的类型。这里是移动操作。
        event.accept()


app = QtWidgets.QApplication(sys.argv)
db = DragButton()
db.show()
sys.exit(app.exec_())
