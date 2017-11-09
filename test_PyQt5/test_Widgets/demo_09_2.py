import sys
from PyQt5 import QtWidgets, QtGui


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.resize(500,500)
        self.setWindowTitle("工具栏示例")

        self.exit_menu = QtWidgets.QAction(QtGui.QIcon(r"../images/new_on.jpg"), "退出", self)
        self.exit_menu.setShortcut("Ctrl+Q")
        self.exit_menu.setStatusTip("退出程序")
        self.exit_menu.triggered.connect(QtWidgets.qApp.quit)

        self.toolbar = self.addToolBar("退出")
        self.toolbar.addAction(self.exit_menu)


app = QtWidgets.QApplication(sys.argv)
main_win = MainWindow()
main_win.show()
sys.exit(app.exec_())
