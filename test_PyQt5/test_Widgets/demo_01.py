from PyQt5 import QtWidgets
import sys
#每一个PyQt5程序都需要有一个application对象，application类包含在QtGui模块中。
# sys.argv参数是一个命令行参数列表。
# Python脚本可以从shell中执行，参数可以让我们选择启动脚本的方式。
app = QtWidgets.QApplication(sys.argv)
# QWidget部件是PyQt5中所有用户界面类的父类。
# 这里我们使用没有参数的默认构造函数，
# 它没有继承其它类。我们称没有父类的first_window为一个window。
first_window = QtWidgets.QWidget()
# resize()方法可以改变窗口部件的大小，在这里我们将其设置为250像素宽，150像素高。
first_window.resize(400, 300)
# 这句用来设置窗口部件的标题，该标题将在标题栏中显示。
first_window.setWindowTitle("This is my first program of PyQt5!")
# show()方法将窗口部件显示在屏幕上。
first_window.show()
sys.exit(app.exec_())  # 这是因为exec是Python的关键字，为避免冲突，PyQt使用exec_()替代。
