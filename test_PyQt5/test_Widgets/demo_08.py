# 菜单栏
# 菜单栏是 GUI 程序最明显的组成部分。它由一组位于不同菜单中的命令组成。
# 在控制台程序中，我们必须记住那些晦涩难懂的命令。但在 GUI 程序中，
# 通过菜单栏我们将命令合理的放置在不同的菜单中来降低学习新应用程序的时间开销。

import sys
from PyQt5 import QtWidgets, QtGui


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        """
        首先我们使用 QMainWindow 类的 menuBar()方法创建一个菜单栏。
        然后使用 addMenu()方法添加一个菜单。
        最后我们把动作对象（这里是exit_menu）添加到 file 菜单中。
        """
        super(MainWindow, self).__init__()
        self.resize(300, 300)
        self.setWindowTitle("菜单栏示例")

        exit_menu = QtWidgets.QAction(QtGui.QIcon(r"../images/new_on.jpg"), "退出", self)
        exit_menu.setShortcut("Ctrl+Q")
        exit_menu.setStatusTip("退出程序")
        exit_menu.triggered.connect(QtWidgets.qApp.quit)

        self.statusBar()

        manubar = self.menuBar()
        file = manubar.addMenu("文件")
        file.addAction(exit_menu)


app = QtWidgets.QApplication(sys.argv)
mainwindow = MainWindow()
mainwindow.show()
sys.exit(app.exec_())
