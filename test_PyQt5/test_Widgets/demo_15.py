"""用Esc键退出示例"""

import sys

from PyQt5 import QtWidgets, QtCore


class Escape(QtWidgets.QWidget):
    def __init__(self):
        super(Escape, self).__init__()

        self.setWindowTitle("Esc退出示例程序")
        self.resize(250, 150)

    def keyPressEvent(self, event):
        """
        重新实现了 keyPressEvent()事件处理方法。
        """
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()


app = QtWidgets.QApplication(sys.argv)
esc = Escape()
esc.show()
sys.exit(app.exec_())
