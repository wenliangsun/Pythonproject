import sys, os

from PyQt5 import QtWidgets, QtCore, QtGui
from PIL import Image
from test_tensorflow.test_demos.demo_06_mnist_softmax_regression import model_run


class RoadDetect(QtWidgets.QMainWindow):
    def __init__(self):
        super(RoadDetect, self).__init__()
        # 设置标题背景
        self.setWindowTitle("demo01")
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.setGeometry(10, 10, screen.width() - 30, screen.height() - 30)
        self.background = QtGui.QImage(r"../images/bj.jpg")
        self.setAutoFillBackground(True)

        # 设置程序图标
        self.setWindowIcon(QtGui.QIcon(r"../images/ipiu.png"))

        # 设置logo和大标题
        self.logo_img = QtWidgets.QLabel(self)
        self.logo_img.setGeometry(25, 5, 180, 60)
        img = QtGui.QPixmap(r"../images/logo_big.png")
        img = img.scaled(self.logo_img.size())
        self.logo_img.setPixmap(img)
        self.title_text = "遥{0}感{0}影{0}像{0}道{0}路{0}检{0}测{0}系{0}统".format(" " * 2)
        self.title = QtWidgets.QLabel(self)
        self.title.setGeometry(220, 5, self.width() - 230, 60)
        self.title.setText(self.title_text)

        self.title.setStyleSheet("QLabel{"
                                 "color:yellow;"
                                 "font-size:48px;"
                                 "font-weight:bold;"
                                 "font-family:宋体}")

        # 设置界线
        self.bd_img = QtWidgets.QLabel(self)
        self.bd_img.setGeometry(0, 70, self.width(), 6)
        img = QtGui.QPixmap(r"../images/bd2.png")
        img = img.scaled(self.bd_img.size())
        self.bd_img.setPixmap(img)

        # 开始绘制
        self.board = Board()
        self.setCentralWidget(self.board)

    def closeEvent(self, event: QtGui.QCloseEvent):
        reply = QtWidgets.QMessageBox.question(self, "确认退出", "你确定要退出么？",
                                               QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        # 重写resizeEvent, 使背景图案可以根据窗口大小改变
        QtWidgets.QWidget.resizeEvent(self, event)
        palette = QtGui.QPalette()
        palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(self.background.scaled(event.size())))
        self.setPalette(palette)


class Board(QtWidgets.QFrame):
    def __init__(self):
        super(Board, self).__init__()
        self.init_UI()
        with open(r"./QSS/roaddetect.css", 'r') as q:
            self.setStyleSheet(q.read())

    def init_UI(self):
        screen = QtWidgets.QDesktopWidget().screenGeometry()
        self.mw_width = screen.width() - 30  # 1336
        self.mw_height = screen.height() - 30  # 738

        # 按钮 和显示
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.open_button = QtWidgets.QPushButton("Open Image", self)
        self.open_button.clicked.connect(self.open_image)

        self.save_button = QtWidgets.QPushButton("Save Image", self)
        self.save_button.clicked.connect(self.save_image)

        self.save_index_button = QtWidgets.QPushButton("Save Index", self)
        self.save_index_button.clicked.connect(self.save_index)

        self.cnn_button = QtWidgets.QPushButton("CNN", self)
        self.cnn_button.setObjectName("cnn_but")
        self.cnn_button.clicked.connect(self.CNN_model)
        self.fcn_button = QtWidgets.QPushButton("FCN", self)
        self.fcn_button.setObjectName("fcn_but")
        self.fcn_button.clicked.connect(self.FCN_model)
        self.gan_button = QtWidgets.QPushButton("GAN", self)
        self.gan_button.setObjectName("gan_but")
        self.gan_button.clicked.connect(self.GAN_model)

        button_ground = QtWidgets.QWidget(self)
        button_ground.setGeometry(int(self.mw_width * 0.01),
                                  int(self.mw_height * 0.15),
                                  int(self.mw_width * 0.25),
                                  int(self.mw_height * 0.16))
        button_grid = QtWidgets.QGridLayout()
        button_grid.addWidget(self.open_button, 0, 0)
        button_grid.addWidget(self.cnn_button, 0, 1)
        button_grid.addWidget(self.save_button, 1, 0)
        button_grid.addWidget(self.fcn_button, 1, 1)
        button_grid.addWidget(self.save_index_button, 2, 0)
        button_grid.addWidget(self.gan_button, 2, 1)
        button_ground.setLayout(button_grid)

        # 显示图像名称以及尺寸
        name_size = QtWidgets.QWidget(self)
        name_size.setGeometry(int(self.mw_width * 0.01),
                              int(self.mw_height * 0.83),
                              int(self.mw_width * 0.24),
                              int(self.mw_height * 0.11))
        name_button = QtWidgets.QPushButton("Image Name", self)
        size_button = QtWidgets.QPushButton("Image Size", self)
        self.name_text = QtWidgets.QLineEdit("None", self)
        self.size_text = QtWidgets.QLineEdit("None", self)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(name_button, 1, 0)
        grid.addWidget(self.name_text, 1, 1)
        grid.addWidget(size_button, 2, 0)
        grid.addWidget(self.size_text, 2, 1)
        name_size.setLayout(grid)

        # 显示评价指标
        evaluate_ground = QtWidgets.QWidget(self)
        evaluate_x = self.mw_width - int(self.mw_width * 0.02) - int(self.mw_width * 0.25)
        evaluate_ground.setGeometry(evaluate_x, int(self.mw_height * 0.15),
                                    int(self.mw_width * 0.25),
                                    int(self.mw_height * 0.16))
        precision_button = QtWidgets.QPushButton("Precision", self)
        self.precision_text = QtWidgets.QLineEdit("None", self)
        recall_button = QtWidgets.QPushButton("Recall", self)
        self.recall_text = QtWidgets.QLineEdit("None", self)
        thresh_button = QtWidgets.QPushButton("Threshold", self)
        self.thresh_text = QtWidgets.QLineEdit("None", self)

        eval_grid = QtWidgets.QGridLayout()
        eval_grid.addWidget(precision_button, 1, 0)
        eval_grid.addWidget(self.precision_text, 1, 1)
        eval_grid.addWidget(recall_button, 2, 0)
        eval_grid.addWidget(self.recall_text, 2, 1)
        eval_grid.addWidget(thresh_button, 3, 0)
        eval_grid.addWidget(self.thresh_text, 3, 1)
        evaluate_ground.setLayout(eval_grid)

        self.message = QtWidgets.QTextEdit(self)
        self.terminal_text = QtWidgets.QLabel(self)
        self.terminal_text.setText("终端信息")
        self.message.setGeometry(evaluate_x,
                                 int(self.mw_height * 0.35),
                                 int(self.mw_width * 0.24),
                                 int(self.mw_height * 0.45))
        self.terminal_text.move(evaluate_x + int(self.mw_width * 0.24) // 2 - 25,
                                int(self.mw_height * 0.35) - 25)

        # 原始图像窗口
        self.img_text = QtWidgets.QLabel(self)
        self.img_text.setText("原始图像")
        self.row_img = QtWidgets.QLabel(self)
        self.row_img.setObjectName("show_pic")
        self.row_img.setGeometry(int(self.mw_width * 0.01),
                                 int(self.mw_height * 0.35),
                                 int(self.mw_width * 0.25),
                                 int(self.mw_height * 0.48))
        img = QtGui.QPixmap(r"../images/26.jpg")
        img = img.scaled(self.row_img.size())
        self.row_img.setPixmap(img)
        self.img_text.move(self.row_img.width() // 2 - 20,
                           int(self.mw_height * 0.35) - 30)

        # 模型输出图
        self.out_text = QtWidgets.QLabel(self)
        self.out_text.setText("预测图像")
        self.out_pred = QtWidgets.QLabel(self)
        self.out_pred.setObjectName("show_pic")
        self.out_pred.setGeometry((self.mw_width - int(self.mw_width * 0.45)) // 2,
                                  (self.mw_height - int(self.mw_height * 0.81)) // 2 + 40,
                                  int(self.mw_width * 0.45),
                                  int(self.mw_height * 0.81))
        # TODO 后面需要进一步修改，目前只是在测试
        pred = QtGui.QPixmap(r"../images/26.jpg")
        pred = pred.scaled(self.out_pred.size())
        self.out_pred.setPixmap(pred)
        self.out_text.move(self.mw_width // 2 - 30,
                           int(self.mw_height * 0.115))

    def open_image(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开文件")
        try:
            img_name = file_name[0].split("/")[-1]
            img = QtGui.QPixmap(file_name[0])
            print(img.width())
            img_size = "{} X {} pixel".format(img.width(), img.height())
        except FileNotFoundError:
            img = ""
            img_name = None
            img_size = None
        img = img.scaled(self.out_pred.size())
        self.out_pred.setPixmap(img)
        img = img.scaled(self.row_img.size())
        self.row_img.setPixmap(img)
        self.name_text.setText(img_name)
        self.size_text.setText(img_size)

    def save_image(self):
        file_path = QtWidgets.QFileDialog.getSaveFileName(self, "保存文件")
        img = Image.fromarray(self.out_pred)
        img.save(file_path[0])
        print("保存图片")

    def save_index(self):
        file_path = QtWidgets.QFileDialog.getSaveFileName(self, "保存指标")
        index = "Image Name: {},\nImage Size: {},\nPrecision: {},\nRecall: {},\nThreshold: {}".format(
            self.name_text.text(), self.size_text.text(), self.precision_text.text(),
            self.recall_text.text(), self.thresh_text.text())
        f = open(file_path[0], 'w')
        f.write(index)
        f.close()

    # TODO 待填###############################
    def CNN_model(self):
        print("CNN model")
        # model_run()
        # data = os.popen("python3 ../../test_tensorflow/test_Widgets/demo_06_mnist_softmax_regression.py", 'r')
        # data = os.popen("python3 ../../test_tensorflow/test_Widgets/demo_07_mnist_CNN.py", 'r')
        data = os.popen("python3 ../../test_tensorflow/test_Widgets/demo_07_mnist_CNN.py", 'r')
        self.message.setText(data.read())

    def FCN_model(self):
        print("FCN model", file=open("aaa.txt", 'w'))
        self.precision_text.setText("1000")
        self.thresh_text.setText("0.8")
        self.recall_text.setText("500")
        img = QtGui.QPixmap(r"../images/15.tif")
        img = img.scaled(self.out_pred.size())
        self.out_pred.setPixmap(img)
        f = open("aaa.txt")
        data = f.read()
        self.message.setText(data)

    def GAN_model(self):
        print("GAN model")


app = QtWidgets.QApplication(sys.argv)
mw = RoadDetect()
mw.show()
sys.exit(app.exec_())
