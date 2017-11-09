import sys

from PyQt5 import QtWidgets, QtGui


class Main(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Main, self).__init__(parent)

        self.setObjectName("main")
        self.setWindowTitle("Hello QT")
        self.setWindowIcon(QtGui.QIcon(r"../images/new_on.jpg"))
        self.resize(300, 300)
        # self.setStyleSheet("QWidget#main{background-color:yellow}")
        with open("main.css", 'r') as q:
            self.setStyleSheet(q.read())


class Main2(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Main2, self).__init__(parent)
        self.setObjectName("main")
        self.resize(300, 300)
        self.setWindowTitle("demo_02")
        self.setWindowIcon(QtGui.QIcon(r"../images/ipiu.png"))

        with open("main.css", 'r') as q:
            self.setStyleSheet(q.read())


class Main3(QtWidgets.QFrame):
    def __init__(self,parent=None):
        super(Main3, self).__init__(parent)
        self.setObjectName("main")
        self.setWindowTitle("Hello")
        self.setWindowIcon(QtGui.QIcon(r"../images/ipiu.png"))
        self.resize(500,500)

        with open("main.css",'r') as q:
            self.setStyleSheet(q.read())


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    main = Main3()
    main.show()
    sys.exit(app.exec_())
