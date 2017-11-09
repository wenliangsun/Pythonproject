import sys, os

from PIL import Image
from PyQt5.QtCore import pyqtSlot
from PyQt5 import QtWidgets, QtGui, QtCore
from mainwindow import Ui_MainWindow
from dialog import Ui_cnn_info, Ui_fcn_info, Ui_gan_info


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        QtCore.QThread.sleep(2)

    @pyqtSlot()
    def on_open_button_clicked(self):
        file_name = QtWidgets.QFileDialog.getOpenFileName(
            self, "打开文件")
        if file_name[0] == '':
            QtWidgets.QMessageBox.warning(self, "警告信息", "打开失败，请正确打开图片！")
        else:
            try:
                img_name = file_name[0].split("/")[-1]
                img_left = QtGui.QPixmap(file_name[0])
                img_mid = img_left
                if not img_left.width():
                    QtWidgets.QMessageBox.warning(self, "警告信息", "打开失败，请正确打开图片！")
                    img_left = QtGui.QPixmap(":/my_pic/remote_1.jpg")
                    img_mid = QtGui.QPixmap(":/my_pic/remote_2.jpg")
                img_size = "{} X {} pixel".format(img_left.width(), img_left.height())
            except FileNotFoundError:
                img_left = QtGui.QPixmap(":/my_pic/remote_1.jpg")
                img_mid = QtGui.QPixmap(":/my_pic/remote_2.jpg")
                img_name = None
                img_size = None
            self.row_img_Label.setText("原始图像")
            img = img_mid.scaled(self.out_img_Label_2.size())
            self.out_img_Label_2.setPixmap(img)
            img = img_left.scaled(self.row_img_Label_2.size())
            self.row_img_Label_2.setPixmap(img)
            self.name_LineEdit.setText(img_name)
            self.size_LineEdit.setText(img_size)

            msg = "Open image...\nimage name: {}\nimage size: {}".format(img_name, img_size)
            self.message_TextEdit.append(msg)

    @pyqtSlot()
    def on_save_index_button_clicked(self):
        file_path = QtWidgets.QFileDialog.getSaveFileName(self, "保存指标")
        index = "Image Name: {},\nImage Size: {},\nPrecision: {},\nRecall: {},\nThreshold: {}".format(
            self.name_LineEdit.text(), self.size_LineEdit.text(), self.precision_LineEdit.text(),
            self.recall_LineEdit.text(), self.thresh_LineEdit.text())
        if file_path[0] != '':
            f = None
            try:
                f = open(file_path[0], 'w')
                f.write(index)
                f.close()
                QtWidgets.QMessageBox.information(self, "提示信息", "保存成功！")
            except FileNotFoundError:
                f.close()

    @pyqtSlot()
    def on_save_image_button_clicked(self):
        file_path = QtWidgets.QFileDialog.getSaveFileName(self, "保存文件")
        if file_path[0] != '':
            img = Image.fromarray(self.out_img_Label_2)
            img.save(file_path[0])
            QtWidgets.QMessageBox.information(self, "提示信息", "保存成功！")

    # TODO 待填###############################
    @pyqtSlot()
    def on_cnn_button_clicked(self):
        self.out_img_Label.setText("       ")
        cnn_params = CnnDialog()
        cnn_params.exec_()
        self.message_TextEdit.append("CNN model selected.")
        if not cnn_params.isCancel:
            params_msg = "CNN model params:\nParam_1: {}\nParam_2: {}\nParam_3: {}\nParam_4: {}".format(
                cnn_params.param_1, cnn_params.param_2,
                cnn_params.param_3, cnn_params.param_4)
            self.message_TextEdit.append(params_msg)

    @pyqtSlot()
    def on_fcn_button_clicked(self):
        self.out_img_Label.setText("       ")
        fcn_params = FcnDialog()
        fcn_params.exec_()
        self.message_TextEdit.append("FCN model selected.")
        if not fcn_params.isCancel:
            params_msg = "FCN model params:\nParam_1: {}\nParam_2: {}\nParam_3: {}\nParam_4: {}".format(
                fcn_params.param_1, fcn_params.param_2,
                fcn_params.param_3, fcn_params.param_4)
            self.message_TextEdit.append(params_msg)

    @pyqtSlot()
    def on_gan_button_clicked(self):
        self.out_img_Label.setText("       ")
        gan_params = GanDialog()
        gan_params.exec_()
        self.message_TextEdit.append("GAN model selected.")
        if not gan_params.isCancel:
            params_msg = "GAN model params:\nParam_1: {}\nParam_2: {}\nParam_3: {}\nParam_4: {}".format(
                gan_params.param_1, gan_params.param_2,
                gan_params.param_3, gan_params.param_4)
            self.message_TextEdit.append(params_msg)

    @pyqtSlot()
    def on_run_button_clicked(self):
        reply = QtWidgets.QMessageBox.question(self, "确认运行", "你确定要运行么？",
                                               QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.message_TextEdit.append("model running...")
            self.out_img_Label.setText("预测图像")
            if self.cnn_button.isChecked():
                # TODO CNN MODEL
                print("run cnn model")
                # model_run()
                # data = os.popen("python3 ../../test_tensorflow/test_Widgets/demo_06_mnist_softmax_regression.py", 'r')
                # data = os.popen("python3 ../../test_tensorflow/test_Widgets/demo_07_mnist_CNN.py", 'r')
                # data = os.popen("python3 ../../test_tensorflow/test_demos/demo_07_mnist_CNN.py", 'r')
                # self.message_TextEdit.append(data.read())
            elif self.fcn_button.isChecked():
                # TODO FCN MODEL
                self.precision_LineEdit.setText("1000")
                self.thresh_LineEdit.setText("0.8")
                self.recall_LineEdit.setText("500")
                img = QtGui.QPixmap(r":/my_pic/remote_3.jpg")
                img = img.scaled(self.out_img_Label_2.size())
                self.out_img_Label_2.setPixmap(img)
            elif self.gan_button.isChecked():
                # TODO GAN MODEL
                print("run gan model")
            else:
                raise Exception("Please select model!!!")
            self.message_TextEdit.append("model run completed.")
            QtWidgets.QMessageBox.information(self, "提示信息", "运行结束！")

    def closeEvent(self, event: QtGui.QCloseEvent):
        self.reply_MessageBox = QtWidgets.QMessageBox()
        self.reply_MessageBox.setObjectName("reply_MessageBox")
        reply = self.reply_MessageBox.question(self, "确认退出", "你确定要退出么？",
                                               QtWidgets.QMessageBox.Yes,
                                               QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        QtWidgets.QWidget.resizeEvent(self, event)
        palette = QtGui.QPalette()
        palette.setBrush(QtGui.QPalette.Window, QtGui.QBrush(self.background.scaled(event.size())))
        self.setPalette(palette)


class CnnDialog(QtWidgets.QDialog, Ui_cnn_info):
    def __init__(self, parent=None):
        super(CnnDialog, self).__init__(parent)
        self.setupUi(self)
        self.isCancel = None
        with open(r"/home/sunwl/Projects/Pythonproject/road_detect_project/show_tools/QSS/dialog.css", 'r') as q:
            self.setStyleSheet(q.read())

    @pyqtSlot()
    def on_ok_button_clicked(self):
        self.param_1 = self.lineEdit.text()
        self.param_2 = self.lineEdit_2.text()
        self.param_3 = self.lineEdit_3.text()
        self.param_4 = self.lineEdit_4.text()
        if self.param_1 == '' or self.param_2 == '' or self.param_3 == '' or self.param_4 == '':
            QtWidgets.QMessageBox.warning(self, "警告信息", "请填写完整！")
        else:
            self.isCancel = False
            self.close()

    @pyqtSlot()
    def on_cancel_button_clicked(self):
        self.isCancel = True
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self.isCancel is None:
            self.isCancel = True
            event.accept()
        else:
            event.accept()


class FcnDialog(QtWidgets.QDialog, Ui_fcn_info):
    def __init__(self, parent=None):
        super(FcnDialog, self).__init__(parent)
        self.isCancel = None
        self.setupUi(self)
        with open(r"/home/sunwl/Projects/Pythonproject/road_detect_project/show_tools/QSS/dialog.css", 'r') as q:
            self.setStyleSheet(q.read())

    @pyqtSlot()
    def on_ok_button_clicked(self):
        self.param_1 = self.lineEdit.text()
        self.param_2 = self.lineEdit_2.text()
        self.param_3 = self.lineEdit_3.text()
        self.param_4 = self.lineEdit_4.text()
        if self.param_1 == '' or self.param_2 == '' or self.param_3 == '' or self.param_4 == '':
            QtWidgets.QMessageBox.warning(self, "警告信息", "请填写完整！")
        else:
            self.isCancel = False
            self.close()

    @pyqtSlot()
    def on_cancel_button_clicked(self):
        self.isCancel = True
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self.isCancel is None:
            self.isCancel = True
            event.accept()
        else:
            event.accept()


class GanDialog(QtWidgets.QDialog, Ui_gan_info):
    def __init__(self, parent=None):
        super(GanDialog, self).__init__(parent)
        self.isCancel = None
        self.setupUi(self)
        with open(r"/home/sunwl/Projects/Pythonproject/road_detect_project/show_tools/QSS/dialog.css", 'r') as q:
            self.setStyleSheet(q.read())

    @pyqtSlot()
    def on_ok_button_clicked(self):
        self.param_1 = self.lineEdit.text()
        self.param_2 = self.lineEdit_2.text()
        self.param_3 = self.lineEdit_3.text()
        self.param_4 = self.lineEdit_4.text()
        if self.param_1 == '' or self.param_2 == '' or self.param_3 == '' or self.param_4 == '':
            QtWidgets.QMessageBox.warning(self, "警告信息", "请填写完整！")
        else:
            self.isCancel = False
            self.close()

    @pyqtSlot()
    def on_cancel_button_clicked(self):
        self.isCancel = True
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent):
        if self.isCancel is None:
            self.isCancel = True
            event.accept()
        else:
            event.accept()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    splash = QtWidgets.QSplashScreen(QtGui.QPixmap(":/my_pic/logo_start.png"))
    splash.show()
    splash.showMessage("正在加载图片资源......", QtCore.Qt.AlignBottom)
    QtCore.QThread.sleep(2)
    splash.showMessage("正在加载配置文件......", QtCore.Qt.AlignBottom)
    QtCore.QThread.sleep(2)
    splash.showMessage("正在渲染界面......", QtCore.Qt.AlignBottom)
    QtCore.QThread.sleep(2)
    app.processEvents()
    ui = MainWindow()
    ui.show()
    splash.finish(ui)
    sys.exit(app.exec_())
