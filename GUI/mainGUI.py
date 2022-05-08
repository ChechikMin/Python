
from PyQt5 import QtWidgets
from mydesign import Ui_MainWindow  # импорт нашего сгенерированного файла
from mydesign1 import Ui_MainWindow1
from mydesign3 import Ui_MainWindow3
import sys


class mywindow(QtWidgets.QMainWindow):

    __hashData = {}
    def __init__(self):
        super(mywindow, self).__init__()
        self.startMainWindow()

    def startMainWindow(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.btnClicked)

    def btnClicked(self):
        #self.window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow1()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.saveData1)
        self.ui.pushButton_2.clicked.connect(self.saveData2)
        self.ui.pushButton_3.clicked.connect(self.saveData3)
        self.ui.pushButton_4.clicked.connect(self.saveData4)
        self.ui.pushButton_5.clicked.connect(self.nextTip)
        self.ui.lineEdit.adjustSize()
    def saveData1(self):
        text = self.ui.lineEdit.text()
        self.__hashData["Age"] = text
        #self.ui.pushButton.setText(text)

    def saveData2(self):
        text = self.ui.lineEdit_2.text()
        self.__hashData["Sex"] = text
        #self.ui.pushButton_2.setText(text)

    def saveData3(self):
        text = self.ui.lineEdit_3.text()
        self.__hashData["Education"] = text
        #self.ui.pushButton_3.setText(text)

    def saveData4(self):
        text = self.ui.lineEdit_4.text()
        self.__hashData["Marrige"] = text
        #self.ui.pushButton_4.setText(text)

    def nextTip(self):
        #тут нужно вызвать нейронку
        # и
        # попросить её посчитать
        # нашего клиента


        # данные для работы
        # с входными данными
        # лежат в __hashData



        self.ui = Ui_MainWindow3()
        self.ui.setupUi(self)

app = QtWidgets.QApplication([])
application = mywindow()
application.show()

sys.exit(app.exec())