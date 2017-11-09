"""显示提示信息"""
import sys
from PyQt5 import QtWidgets, QtCore, QtGui


class Tooltip(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(Tooltip, self).__init__()
        # QtWidgets.QWidget.__init__(self,parent)

        self.setGeometry(200, 200, 250, 150)
        self.setWindowTitle("Tip massage")

        self.setToolTip("This is a <b>Qwidget<b> widget")
        QtWidgets.QToolTip.setFont(QtGui.QFont("Times", 10))


app = QtWidgets.QApplication(sys.argv)
tooltip = Tooltip()
tooltip.show()
sys.exit(app.exec_())
