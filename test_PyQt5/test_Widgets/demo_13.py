"""
网格布局
最通用的布局类别是网格布局（QGridLayout）。
该布局方式将窗口空间划分为许多行和列。
要创建该布局方式，我们需要使用 QGridLayout 类。
"""

import sys
from PyQt5 import QtWidgets


class GridLayout(QtWidgets.QMainWindow):
    def __init__(self):
        super(GridLayout, self).__init__()

        self.setWindowTitle("网格布局演示示例")
        buttons_name = ["Cls", "Bck", '', "Close",
                        '7', '8', '9', '/',
                        '5', '6', '7', '*',
                        '1', '2', '3', '-',
                        '0', '.', '=', '+']
        main_ground = QtWidgets.QWidget()
        self.setCentralWidget(main_ground)  # 将它置为中心部件
        grid = QtWidgets.QGridLayout()  # 创建一个网格布局

        for [n, (x, y)] in enumerate([i, j] for i in range(5) for j in range(4)):
            if (x, y) == (0, 2):
                # 使用 addWidget()方法，我们将部件加入到网格布局中。
                # addWidget()方法的参数依次为要加入到局部的部件，行号和列号。
                grid.addWidget(QtWidgets.QLabel(buttons_name[n]), x, y)
            else:
                grid.addWidget(QtWidgets.QPushButton(buttons_name[n]), x, y)

        main_ground.setLayout(grid)
        """
        最后将创建好的网格布局用setLayout方法置于之前创建的main_ground实例上，
        如果不创建main_ground而是直接在最后用self.setLayout(grid)，
        程序可以正常跑起来，但是你看不到你布置好的网格布局，
        同时会被警告QWidget::setLayout: Attempting to set QLayout "" onGridLayout "", 
        which already has a layout。在4.2中我们直接应用self.setLayout(grid)
        方法将布局安放在了程序界面，这是因为我们的主程序界面继承的是
        QtWidgets.QWidget类而不是这节中的QtWidgets.QMainWindow类。
        """


app = QtWidgets.QApplication(sys.argv)
grid_layout = GridLayout()
grid_layout.show()
sys.exit(app.exec_())
