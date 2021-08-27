# 运行程序需安装：tensorflow、numpy、pillow、translate、matplotlib、pyqt5、OpenCV-python
# TensorFlow、pillow安装可用文件中文件，pip install Pillow-8.3.1-cp36-cp36m-win_amd64.whl
#                                     pip install tensorflow-1.4.0-cp36-cp36m-win_amd64.whl

from PyQt5 import QtCore, QtGui, QtWidgets
from sys import argv, exit
from PyQt5.QtWidgets import QApplication, QMainWindow
import icon_rc
import time
import cv2

class Ui_MainWindow(object):
    def __init__(self, MainWindow):

        self.timer_camera = QtCore.QTimer() # 定时器
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)
        self.cap = cv2.VideoCapture() # 准备获取图像
        self.CAM_NUM = 0

        self.slot_init() # 设置槽函数

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(765, 645)
        MainWindow.setMinimumSize(QtCore.QSize(765, 645))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/pic/pai.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolTip("")
        MainWindow.setAutoFillBackground(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_face = QtWidgets.QLabel(self.centralwidget)
        self.label_face.setGeometry(QtCore.QRect(90, 130, 571, 291))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_face.sizePolicy().hasHeightForWidth())
        self.label_face.setSizePolicy(sizePolicy)
        self.label_face.setMinimumSize(QtCore.QSize(0, 0))
        self.label_face.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(16)
        self.label_face.setFont(font)
        self.label_face.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_face.setStyleSheet("background-color: rgb(192, 218, 255);")
        self.label_face.setAlignment(QtCore.Qt.AlignCenter)
        self.label_face.setObjectName("label_face")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 731, 41))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("华文隶书")
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.pushButton_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open.setGeometry(QtCore.QRect(110, 80, 120, 40))
        self.pushButton_open.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_open.setMaximumSize(QtCore.QSize(120, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_open.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/pic/g1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_open.setIcon(icon1)
        self.pushButton_open.setObjectName("pushButton_open")
        self.pushButton_take = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_take.setGeometry(QtCore.QRect(260, 80, 100, 40))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_take.sizePolicy().hasHeightForWidth())
        self.pushButton_take.setSizePolicy(sizePolicy)
        self.pushButton_take.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_take.setMaximumSize(QtCore.QSize(100, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_take.setFont(font)
        self.pushButton_take.setIcon(icon)
        self.pushButton_take.setObjectName("pushButton_take")
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close.setGeometry(QtCore.QRect(520, 80, 120, 40))
        self.pushButton_close.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_close.setMaximumSize(QtCore.QSize(130, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_close.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/pic/down.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_close.setIcon(icon2)
        self.pushButton_close.setObjectName("pushButton_close")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(390, 80, 93, 41))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(100, 130, 551, 421))
        self.label_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_2.setText("")
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(110, 480, 91, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑 Light")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(240, 480, 231, 41))
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        self.pushButton_2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_2.setGeometry(QtCore.QRect(100, 560, 101, 41))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.label_face.raise_()
        self.label.raise_()
        self.pushButton_open.raise_()
        self.pushButton_take.raise_()
        self.pushButton_close.raise_()
        self.pushButton.raise_()
        self.label_3.raise_()
        self.label_4.raise_()
        self.pushButton_2.raise_()
        self.label_2.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.actionGoogle_Translate = QtWidgets.QAction(MainWindow)
        self.actionGoogle_Translate.setObjectName("actionGoogle_Translate")
        self.actionHTML_type = QtWidgets.QAction(MainWindow)
        self.actionHTML_type.setObjectName("actionHTML_type")
        self.actionsoftware_version = QtWidgets.QAction(MainWindow)
        self.actionsoftware_version.setObjectName("actionsoftware_version")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Camera Recognition"))
        self.label_face.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>"))
        self.label.setText(_translate("MainWindow", "摄像头识别"))
        self.pushButton_open.setToolTip(_translate("MainWindow", "点击打开摄像头"))
        self.pushButton_open.setText(_translate("MainWindow", "打开摄像头"))
        self.pushButton_take.setToolTip(_translate("MainWindow", "点击拍照"))
        self.pushButton_take.setText(_translate("MainWindow", "拍照"))
        self.pushButton_close.setToolTip(_translate("MainWindow", "点击关闭摄像头"))
        self.pushButton_close.setText(_translate("MainWindow", "关闭摄像头"))
        self.pushButton.setText(_translate("MainWindow", "识别"))
        self.label_3.setText(_translate("MainWindow", "识别结果："))
        self.pushButton_2.setText(_translate("MainWindow", "打开图表"))
        self.actionGoogle_Translate.setText(_translate("MainWindow", "Google Translate"))
        self.actionHTML_type.setText(_translate("MainWindow", "HTML type"))
        self.actionsoftware_version.setText(_translate("MainWindow", "software version"))

    def slot_init(self):
        # 设置槽函数
        self.pushButton_open.clicked.connect(self.button_open_camera_click)
        self.timer_camera.timeout.connect(self.show_camera)
        self.pushButton_close.clicked.connect(self.closeEvent)
        self.pushButton_take.clicked.connect(self.takePhoto)
        self.pushButton.clicked.connect(self.outcome)
        self.pushButton_2.clicked.connect(self.chart)


    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"请检测相机与电脑是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(10)



    def show_camera(self):
        flag, self.image = self.cap.read()

        self.image=cv2.flip(self.image, 1) # 左右翻转
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
        self.label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label_face.setScaledContents(True)


    def takePhoto(self):
        if self.timer_camera.isActive() != False:
            now_time = time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))
            print(now_time)
            cv2.imwrite('pic'+'.png', self.image)

            cv2.putText(self.image, 'The picture have saved !',
                        (int(self.image.shape[1]/2-130), int(self.image.shape[0]/2)),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        1.0, (255, 0, 0), 1)

            self.timer_camera.stop()

            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 左右翻转

            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.label_face.setScaledContents(True)



    def closeEvent(self):
        if self.timer_camera.isActive() != False:
            ok = QtWidgets.QPushButton()
            cacel = QtWidgets.QPushButton()

            msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")

            msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
            msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
            ok.setText(u'确定')
            cacel.setText(u'取消')

            if msg.exec_() != QtWidgets.QMessageBox.RejectRole:

                if self.cap.isOpened():
                    self.cap.release()
                if self.timer_camera.isActive():
                    self.timer_camera.stop()
                self.label_face.setText("<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")


    def outcome(self):
        from translate import Translator

        translator = Translator(to_lang="chinese")

        import tensorflow as tf
        import numpy as np
        import re
        import os

        model_dir = 'model/'
        image = 'pic.png'

        # 将类别ID转换为人类易读的标签
        class NodeLookup(object):
            def __init__(self,
                         label_lookup_path=None,
                         uid_lookup_path=None):
                if not label_lookup_path:
                    label_lookup_path = os.path.join(
                        model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
                if not uid_lookup_path:
                    uid_lookup_path = os.path.join(
                        model_dir, 'imagenet_synset_to_human_label_map.txt')
                self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

            def load(self, label_lookup_path, uid_lookup_path):
                if not tf.gfile.Exists(uid_lookup_path):
                    tf.logging.fatal('File does not exist %s', uid_lookup_path)
                if not tf.gfile.Exists(label_lookup_path):
                    tf.logging.fatal('File does not exist %s', label_lookup_path)

                # Loads mapping from string UID to human-readable string
                proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
                uid_to_human = {}
                p = re.compile(r'[n\d]*[ \S,]*')
                for line in proto_as_ascii_lines:
                    parsed_items = p.findall(line)
                    uid = parsed_items[0]
                    human_string = parsed_items[2]
                    uid_to_human[uid] = human_string

                # Loads mapping from string UID to integer node ID.
                node_id_to_uid = {}
                proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
                for line in proto_as_ascii:
                    if line.startswith('  target_class:'):
                        target_class = int(line.split(': ')[1])
                    if line.startswith('  target_class_string:'):
                        target_class_string = line.split(': ')[1]
                        node_id_to_uid[target_class] = target_class_string[1:-2]

                # Loads the final mapping of integer node ID to human-readable string
                node_id_to_name = {}
                for key, val in node_id_to_uid.items():
                    if val not in uid_to_human:
                        tf.logging.fatal('Failed to locate: %s', val)
                    name = uid_to_human[val]
                    node_id_to_name[key] = name

                return node_id_to_name

            def id_to_string(self, node_id):
                if node_id not in self.node_lookup:
                    return ''
                return self.node_lookup[node_id]

        # 读取训练好的Inception-v3模型来创建graph
        def create_graph():
            with tf.gfile.FastGFile(os.path.join(
                    model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')

        # 读取图片
        image_data = tf.gfile.FastGFile(image, 'rb').read()

        # 创建graph
        create_graph()

        sess = tf.Session()
        # Inception-v3模型的最后一层softmax的输出
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        # 输入图像数据，得到softmax概率值（一个shape=(1,1008)的向量）
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        # (1,1008)->(1008,)
        predictions = np.squeeze(predictions)

        # ID --> English string label.
        node_lookup = NodeLookup()
        # 取出前5个概率最大的值（top-5)
        # %%
        k = 0
        kind = np.array(['a', 'a', 'a', 'a', 'a'], dtype=object)
        probability = np.zeros(5)
        top_5 = predictions.argsort()[-5:][::-1]
        for node_id in top_5:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            probability[k] = '%.4f' % score
            kind[k] = translator.translate(human_string.split(",")[0])
            k = k + 1


        sess.close()
        # %%
        import matplotlib.pyplot as plt

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        plt.bar(kind, probability, width=0.5)
        plt.xlabel('种类')
        plt.ylabel('概率')
        plt.title('识别出的最有可能的5个结果及其概率')
        for x, y in zip(kind, probability):
            plt.text(x, y, str(y), ha='center', va='bottom', fontsize=10.5)
        plt.savefig('chart.jpg')
        if probability[0] <0.8:
            self.label_4.setText("无法识别")
        else:
            self.label_4.setText(kind[0])

    def chart(self):
        _translate = QtCore.QCoreApplication.translate
        if self.pushButton_2.text() == "打开图表":
            showImage = QtGui.QImage('chart.jpg')
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.label_2.setScaledContents(True)
            self.pushButton_2.setText(_translate("MainWindow", "关闭图表"))
        else:
            showImage = QtGui.QImage('')
            self.label_2.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.label_2.setScaledContents(True)
            self.pushButton_2.setText(_translate("MainWindow", "打开图表"))



if __name__ == '__main__':
    app = QApplication(argv)

    window = QMainWindow()
    ui = Ui_MainWindow(window)

    window.show()
    exit(app.exec_())
