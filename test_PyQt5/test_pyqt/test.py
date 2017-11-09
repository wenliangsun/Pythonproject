from test_pyqt import Ui_MainWindow

if __name__ == "__main__":
    import sys
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *

    app = QApplication(sys.argv)
    widget = QMainWindow(None)
    Ui_MainWindow().setupUi(widget)
    widget.show()
    sys.exit(app.exec_())
