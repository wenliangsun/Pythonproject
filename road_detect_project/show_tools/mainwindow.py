import source_file
from PyQt5 import QtWidgets, QtCore, QtGui


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("demo02")
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        MainWindow.setGeometry(15, 15, screen.width() - 30, screen.height() - 30)
        MainWindow.setWindowIcon(QtGui.QIcon(":/my_pic/ipiu.png"))
        MainWindow.setAutoFillBackground(True)

        self.background = QtGui.QImage(":/my_pic/bj.jpg")

        self.centralWidget = QtWidgets.QFrame(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        self.logo_label = QtWidgets.QLabel(self.centralWidget)
        self.logo_label.setObjectName("logo_label")
        self.logo_label.setGeometry(25, 5, 180, 60)
        self.logo_label.setPixmap(QtGui.QPixmap(":/my_pic/logo.png").scaled(self.logo_label.size()))

        self.title_label = QtWidgets.QLabel(self.centralWidget)
        self.title_label.setObjectName("title_label")
        self.title_label.setGeometry(220, 5, screen.width() - 230, 60)
        self.title_label.setText("遥{0}感{0}影{0}像{0}道{0}路{0}检{0}测{0}系{0}统".format(" " * 2))

        self.bound_label = QtWidgets.QLabel(self.centralWidget)
        self.bound_label.setObjectName("bound_label")
        self.bound_label.setGeometry(0, 70, MainWindow.width(), 6)
        self.bound_label.setPixmap(QtGui.QPixmap(":/my_pic/bd.png").scaled(self.bound_label.size()))

        self.gridLayoutWidget = QtWidgets.QWidget(self.centralWidget)
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayoutWidget.setGeometry(int(screen.width() * 0.01),
                                          int(screen.height() * 0.15),
                                          int(screen.width() * 0.255),
                                          int(screen.height() * 0.16))
        self.grid_button = QtWidgets.QGridLayout()
        self.grid_button.setObjectName("grid_button")
        self.open_button = QtWidgets.QPushButton("Open Image", self.centralWidget)
        self.open_button.setObjectName("open_button")
        self.save_image_button = QtWidgets.QPushButton("Save Image", self.centralWidget)
        self.save_image_button.setObjectName("save_image_button")
        self.save_index_button = QtWidgets.QPushButton("Save Index", self.centralWidget)
        self.save_index_button.setObjectName("save_index_button")
        self.cnn_button = QtWidgets.QRadioButton("CNN Model", self.centralWidget)
        self.cnn_button.setObjectName("cnn_button")
        self.cnn_button.setChecked(True)
        self.fcn_button = QtWidgets.QRadioButton("FCN Model", self.centralWidget)
        self.fcn_button.setObjectName("fcn_button")
        self.gan_button = QtWidgets.QRadioButton("GAN Model", self.centralWidget)
        self.gan_button.setObjectName("gan_button")
        self.grid_button.addWidget(self.open_button, 1, 0)
        self.grid_button.addWidget(self.cnn_button, 1, 1)
        self.grid_button.addWidget(self.save_image_button, 2, 0)
        self.grid_button.addWidget(self.fcn_button, 2, 1)
        self.grid_button.addWidget(self.save_index_button, 3, 0)
        self.grid_button.addWidget(self.gan_button, 3, 1)
        self.gridLayoutWidget.setLayout(self.grid_button)

        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralWidget)
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayoutWidget_2.setGeometry(int(screen.width() * 0.01),
                                            int(screen.height() * 0.83),
                                            int(screen.width() * 0.255),
                                            int(screen.height() * 0.11))
        self.grid_img_msg = QtWidgets.QGridLayout()
        self.grid_img_msg.setObjectName("grid_img_msg")
        self.name_button = QtWidgets.QPushButton("Image Name", self.centralWidget)
        self.name_button.setObjectName("name_button")
        self.size_button = QtWidgets.QPushButton("Image Size", self.centralWidget)
        self.size_button.setObjectName("size_button")
        self.name_LineEdit = QtWidgets.QLineEdit('', self.centralWidget)
        self.name_LineEdit.setObjectName("name_LineEdit")
        self.size_LineEdit = QtWidgets.QLineEdit('', self.centralWidget)
        self.size_LineEdit.setObjectName("size_LineEdit")
        self.grid_img_msg.addWidget(self.name_button, 1, 0)
        self.grid_img_msg.addWidget(self.name_LineEdit, 1, 1)
        self.grid_img_msg.addWidget(self.size_button, 2, 0)
        self.grid_img_msg.addWidget(self.size_LineEdit, 2, 1)
        self.gridLayoutWidget_2.setLayout(self.grid_img_msg)

        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.centralWidget)
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        evaluate_x = screen.width() - int(screen.width() * 0.02) - int(screen.width() * 0.255)
        self.gridLayoutWidget_3.setGeometry(evaluate_x, int(screen.height() * 0.15),
                                            int(screen.width() * 0.225),
                                            int(screen.height() * 0.16))
        self.grid_evaluate = QtWidgets.QGridLayout()
        self.grid_evaluate.setObjectName("grid_evaluate")
        self.precision_button = QtWidgets.QPushButton("Precision", self.centralWidget)
        self.precision_button.setObjectName("precision_button")
        self.precision_LineEdit = QtWidgets.QLineEdit('', self.centralWidget)
        self.precision_LineEdit.setObjectName("precision_LineEdit")
        self.recall_button = QtWidgets.QPushButton("Recall", self.centralWidget)
        self.recall_button.setObjectName("recall_button")
        self.recall_LineEdit = QtWidgets.QLineEdit('', self.centralWidget)
        self.recall_LineEdit.setObjectName("recall_LineEdit")
        self.thresh_button = QtWidgets.QPushButton("Threshold", self.centralWidget)
        self.thresh_button.setObjectName("thresh_button")
        self.thresh_LineEdit = QtWidgets.QLineEdit('', self.centralWidget)
        self.thresh_LineEdit.setObjectName("thresh_LineEdit")
        self.grid_evaluate.addWidget(self.precision_button, 1, 0)
        self.grid_evaluate.addWidget(self.precision_LineEdit, 1, 1)
        self.grid_evaluate.addWidget(self.recall_button, 2, 0)
        self.grid_evaluate.addWidget(self.recall_LineEdit, 2, 1)
        self.grid_evaluate.addWidget(self.thresh_button, 3, 0)
        self.grid_evaluate.addWidget(self.thresh_LineEdit, 3, 1)
        self.gridLayoutWidget_3.setLayout(self.grid_evaluate)

        self.message_TextEdit = QtWidgets.QTextEdit(self.centralWidget)
        self.message_TextEdit.setObjectName("message_TextEdit")
        self.terminal_Label = QtWidgets.QLabel(self.centralWidget)
        self.terminal_Label.setObjectName("terminal_Label")
        self.terminal_Label.setText("终端信息")
        self.message_TextEdit.setGeometry(evaluate_x,
                                          int(screen.height() * 0.35),
                                          int(screen.width() * 0.225),
                                          int(screen.height() * 0.45))
        self.terminal_Label.move(evaluate_x + int(screen.width() * 0.24) // 2 - 25,
                                 int(screen.height() * 0.35) - 25)

        self.row_img_Label = QtWidgets.QLabel(self.centralWidget)
        self.row_img_Label.setObjectName("row_img_Label")
        self.row_img_Label.setText('         ')
        self.row_img_Label_2 = QtWidgets.QLabel(self.centralWidget)
        self.row_img_Label_2.setObjectName("row_img_Label_2")
        self.row_img_Label_2.setGeometry(int(screen.width() * 0.01),
                                         int(screen.height() * 0.35),
                                         int(screen.width() * 0.25),
                                         int(screen.height() * 0.48))
        self.row_img_Label_2.setPixmap(QtGui.QPixmap(":/my_pic/remote_1.jpg").scaled(self.row_img_Label_2.size()))
        self.row_img_Label.move(self.row_img_Label_2.width() // 2 - 20,
                                int(screen.height() * 0.35) - 30)

        self.out_img_Label = QtWidgets.QLabel(self.centralWidget)
        self.out_img_Label.setObjectName("out_img_Label")
        self.out_img_Label.setText('         ')
        self.run_button = QtWidgets.QPushButton("RUN", self.centralWidget)
        self.run_button.setObjectName("run_button")
        self.run_button.resize(120, 28)
        self.out_img_Label_2 = QtWidgets.QLabel(self.centralWidget)
        self.out_img_Label_2.setObjectName("out_img_Label_2")
        self.out_img_Label_2.setGeometry((screen.width() - int(screen.width() * 0.46)) // 2,
                                         (screen.height() - int(screen.height() * 0.81)) // 2 + 40,
                                         int(screen.width() * 0.45),
                                         int(screen.height() * 0.80))
        # TODO 后面需要进一步修改，目前只是在测试
        self.out_img_Label_2.setPixmap(QtGui.QPixmap(":/my_pic/remote_2.jpg").scaled(self.out_img_Label_2.size()))
        self.run_button.move(screen.width() // 2 - 100,
                             int(screen.height() * 0.108))
        self.out_img_Label.move(screen.width() // 2 + 30,
                                int(screen.height() * 0.112))

        with open(r"/home/sunwl/Projects/Pythonproject/road_detect_project/show_tools/QSS/mainwindow.css", 'r') as q:
            self.centralWidget.setStyleSheet(q.read())

        MainWindow.setCentralWidget(self.centralWidget)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
