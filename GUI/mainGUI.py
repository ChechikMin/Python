import sys
from collections import namedtuple
from typing import Tuple, List

from PyQt5 import QtWidgets

from mydesign import Ui_MainWindow
from mydesign1 import Ui_MainWindow1
from mydesign3 import Ui_MainWindow3
from classification import *



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

        self.ui.comboBox.addItem("Male",1)
        self.ui.comboBox.addItem("Female", 2)

        self.ui.comboBox_2.addItem("Master/GraduateStudent", 1)
        self.ui.comboBox_2.addItem("Bachelour", 2)
        self.ui.comboBox_2.addItem("SecondarySchool/FurtherEd", 3)

        self.ui.comboBox_3.addItem("Married", 1)
        self.ui.comboBox_3.addItem("Single", 2)
        self.ui.comboBox_3.addItem("Other", 3)

        self.ui.buttonBox.accepted.connect(self.saveData)
        self.ui.pushButton_5.clicked.connect(self.nextTipNonLin)
        self.ui.pushButton_6.clicked.connect(self.back)
        self.ui.lineEdit.adjustSize()

    def saveData(self):
        pass

    def back(self):
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.btnClicked)


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
        self.ui.pushButton.clicked.connect(self.back1)

    def nextTipNonLin(self):
        # data = self.prepare_data(self.__hashData)
        #
        # model = Classification(11, 1)
        # model.load_state_dict(torch.load('bin/non_lin-UCI_13_rub.pt'))
        # model.eval()
        #
        # with torch.no_grad():
        #     if torch.cuda.is_available():
        #         inputs = torch.tensor(data, requires_grad=True, dtype=torch.float).cuda()
        #     else:
        #         inputs = torch.tensor(data, requires_grad=True, dtype=torch.float)
        #     output = model.forward(inputs)
        # result = bool(round(output.data.item()))

        self.ui = Ui_MainWindow3()
        self.ui.setupUi(self)
        self.ui.pushButton.clicked.connect(self.back1)

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
        self.ui.pushButton_5.clicked.connect(self.nextTipNonLin)
        self.ui.pushButton_6.clicked.connect(self.back)
        self.ui.lineEdit.adjustSize()


app = QtWidgets.QApplication([])
application = MyWindow()
application.show()

sys.exit(app.exec())
