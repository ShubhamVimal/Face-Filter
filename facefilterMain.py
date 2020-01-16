from PyQt5 import QtCore, QtGui, QtWidgets
from functools import partial
import applyFilters


class Ui_MainWindow(QtWidgets.QMainWindow):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(554, 531)
        MainWindow.setMaximumSize(QtCore.QSize(554, 531))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 270, 511, 25))
        self.layoutWidget.setObjectName("layoutWidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.pushButton_2 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.pushButton_3 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_4 = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_4.setObjectName("pushButton_4")
        self.horizontalLayout.addWidget(self.pushButton_4)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(80, 480, 381, 25))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget1)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_6 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_2.addWidget(self.pushButton_6)
        self.pushButton_7 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_7.setObjectName("pushButton_7")
        self.horizontalLayout_2.addWidget(self.pushButton_7)
        self.pushButton_8 = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout_2.addWidget(self.pushButton_8)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(20, 100, 511, 161))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.layoutWidget2)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.label_2 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_3.addWidget(self.label_2)
        self.label_4 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_3.addWidget(self.label_4)
        self.label_3 = QtWidgets.QLabel(self.layoutWidget2)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_3.addWidget(self.label_3)
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(80, 320, 381, 151))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget3)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_6 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_6.setObjectName("label_6")
        self.horizontalLayout_4.addWidget(self.label_6)
        self.label_7 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_4.addWidget(self.label_7)
        self.label_8 = QtWidgets.QLabel(self.layoutWidget3)
        self.label_8.setObjectName("label_8")
        self.horizontalLayout_4.addWidget(self.label_8)
        self.pushButton_9 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_9.setGeometry(QtCore.QRect(220, 30, 101, 31))
        self.pushButton_9.setStyleSheet("font: 75 10pt \"MS Shell Dlg 2\";")
        self.pushButton_9.setObjectName("pushButton_9")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Use"))
        self.pushButton_2.setText(_translate("MainWindow", "Use"))
        self.pushButton_3.setText(_translate("MainWindow", "Use"))
        self.pushButton_4.setText(_translate("MainWindow", "Use"))
        self.pushButton_6.setText(_translate("MainWindow", "Use"))
        self.pushButton_7.setText(_translate("MainWindow", "Use"))
        self.pushButton_8.setText(_translate("MainWindow", "Use"))
        self.pushButton_9.setText(_translate("MainWindow", "Start Camera"))

        self.label.setText(_translate("MainWindow", "Cat"))
        self.label_2.setText(_translate("MainWindow", "Dog"))
        self.label_4.setText(_translate("MainWindow", "Moustache"))
        self.label_3.setText(_translate("MainWindow", "PigNose"))
        self.label_6.setText(_translate("MainWindow", "Specs"))
        self.label_7.setText(_translate("MainWindow", "WhiteMask"))
        self.label_8.setText(_translate("MainWindow", "PinkMask"))

        self.label.setPixmap(QtGui.QPixmap('Filters/cat.png').scaledToWidth(122))
        self.label_2.setPixmap(QtGui.QPixmap('Filters/dog.png').scaledToWidth(122))
        self.label_4.setPixmap(QtGui.QPixmap('Filters/moustache.png').scaledToWidth(122))
        self.label_3.setPixmap(QtGui.QPixmap('Filters/pig_nose.png').scaledToWidth(122))
        self.label_6.setPixmap(QtGui.QPixmap('Filters/specs_1.png').scaledToWidth(122))
        self.label_7.setPixmap(QtGui.QPixmap('Filters/whiteMask.png').scaledToWidth(122))
        self.label_8.setPixmap(QtGui.QPixmap('Filters/pinkMask.png').scaledToWidth(122))

        self.pushButton_9.clicked.connect(partial(applyFilters.main, 'camera'))
        self.pushButton.clicked.connect(partial(applyFilters.main, 'cat'))
        self.pushButton_2.clicked.connect(partial(applyFilters.main, 'dog'))
        self.pushButton_3.clicked.connect(partial(applyFilters.main, 'moustache'))
        self.pushButton_4.clicked.connect(partial(applyFilters.main, 'pigNose'))
        self.pushButton_6.clicked.connect(partial(applyFilters.main, 'specs'))
        self.pushButton_7.clicked.connect(partial(applyFilters.main, 'whiteMask'))
        self.pushButton_8.clicked.connect(partial(applyFilters.main, 'pinkMask'))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())