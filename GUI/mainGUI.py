from collections import namedtuple
import sys
from typing import Tuple, List

import numpy as np
import pandas as pd
from PyQt5 import QtWidgets

from ..include.classification import Classification
from mydesign import Ui_MainWindow  # импорт нашего сгенерированного файла
from mydesign1 import Ui_MainWindow1
from mydesign3 import Ui_MainWindow3


class MyWindow(QtWidgets.QMainWindow):
    __hashData = {}
    COLUMNS = ['LIMIT_BAL', 'SEX', 'EDUCATION',
               'MARRIAGE', 'AGE', 'PAY_AMT1', 'PAY_AMT2',
               'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
    LIMIT_BAL_CONST = 0.95

    def __init__(self):
        super(MyWindow, self).__init__()
        self.startMainWindow()

    def startMainWindow(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.btnClicked)

    def btnClicked(self):
        # self.window = QtWidgets.QMainWindow()
        self.ui = Ui_MainWindow1()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.saveData1)
        self.ui.pushButton_2.clicked.connect(self.saveData2)
        self.ui.pushButton_3.clicked.connect(self.saveData3)
        self.ui.pushButton_4.clicked.connect(self.saveData4)
        self.ui.pushButton_5.clicked.connect(self.nextTip)
        self.ui.pushButton_6.clicked.connect(self.back)
        self.ui.lineEdit.adjustSize()

    def back(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.btnClicked)

    def saveData1(self):
        text = self.ui.lineEdit.text()
        self.__hashData["Age"] = text
        # self.ui.pushButton.setText(text)

    def saveData2(self):
        text = self.ui.lineEdit_2.text()
        self.__hashData["Sex"] = text
        # self.ui.pushButton_2.setText(text)

    def saveData3(self):
        text = self.ui.lineEdit_3.text()
        self.__hashData["Education"] = text
        # self.ui.pushButton_3.setText(text)

    def saveData4(self):
        text = self.ui.lineEdit_4.text()
        self.__hashData["Marriage"] = text
        # self.ui.pushButton_4.setText(text)

    def saveData5(self):
        text = self.ui.lineEdit_5.text()
        self.__hashData["Credit_sum"] = text
        # self.ui.pushButton_5.setText(text)

    def saveData6(self):
        text = self.ui.lineEdit_6.text()
        self.__hashData["PAY_AMT1"] = text
        # self.ui.pushButton_6.setText(text)

    def saveData7(self):
        text = self.ui.lineEdit_7.text()
        self.__hashData["PAY_AMT2"] = text
        # self.ui.pushButton_7.setText(text)

    def saveData8(self):
        text = self.ui.lineEdit_8.text()
        self.__hashData["PAY_AMT3"] = text
        # self.ui.pushButton_8.setText(text)

    def saveData9(self):
        text = self.ui.lineEdit_9.text()
        self.__hashData["PAY_AMT4"] = text
        # self.ui.pushButton_9.setText(text)

    def saveData10(self):
        text = self.ui.lineEdit_10.text()
        self.__hashData["PAY_AMT5"] = text
        # self.ui.pushButton_10.setText(text)

    def saveData11(self):
        text = self.ui.lineEdit_11.text()
        self.__hashData["PAY_AMT6"] = text
        # self.ui.pushButton_11.setText(text)

    def prepare_data(self, data: dict) -> np.ndarray:
        df = pd.DataFrame(columns=self.COLUMNS)
        for key, val in data.items():
            df[key] = [val]
        return np.array(df)[0]

    def nextTipLogRegr(self):
        data = self.prepare_data(self.__hashData)

        model = Classification(11, 1)
        model.load_state_dict(torch.load('bin/log_regr-UCI_13_rub.pt'))
        model.eval()

        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = torch.tensor(data, requires_grad=True, dtype=torch.float).cuda()
            else:
                inputs = torch.tensor(data, requires_grad=True, dtype=torch.float)
            output = model.forward(inputs)
        result = bool(round(output.data.item()))

        self.ui = Ui_MainWindow3()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.btnClicked)

    def nextTipNonLin(self):
        data = self.prepare_data(self.__hashData)

        model = Classification(11, 1)
        model.load_state_dict(torch.load('bin/non_lin-UCI_13_rub.pt'))
        model.eval()

        with torch.no_grad():
            if torch.cuda.is_available():
                inputs = torch.tensor(data, requires_grad=True, dtype=torch.float).cuda()
            else:
                inputs = torch.tensor(data, requires_grad=True, dtype=torch.float)
            output = model.forward(inputs)
        result = bool(round(output.data.item()))

        self.ui = Ui_MainWindow3()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.btnClicked)

    def getRecommendations(self, pars: torch.nn.parameter.Parameter, data: np.ndarray) -> Tuple[List[str], int]:
        Par = namedtuple('Par', ['val', 'name'])
        par_col = []
        for par, col in zip(list(pars)[-2][0].tolist(), self.COLUMNS):
            par_col.append(Par(par, col))
        par_col.sort(reverse=True)

        # ТОП-3 параметра, которые не позволяют выдать кредит:
        bad_pars = [par.name for par in par_col if par.name != 'SEX'][:3]

        model = Classification(11, 1)
        model.load_state_dict('bin/log_regr-UCI_13_rub.pt')
        model.eval()

        result = False
        while not result:
            data[0] *= self.LIMIT_BAL_CONST
            with torch.no_grad():
                if torch.cuda.is_available():
                    inputs = torch.tensor(data, requires_grad=True, dtype=torch.float).cuda()
                else:
                    inputs = torch.tensor(data, requires_grad=True, dtype=torch.float)
                output = model.forward(inputs)
            result = bool(round(output.data.item()))

        # Сумма, на которую можно выдать кредит с текущими параметрами
        good_sum = round(data[0])

        return bad_pars, good_sum

    def back1(self):
        self.ui = Ui_MainWindow1()
        self.ui.setupUi(self)


app = QtWidgets.QApplication([])
application = MyWindow()
application.show()

sys.exit(app.exec())
