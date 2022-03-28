import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn
import torch.nn.functional as F
from torch.autograd import Variable

from abstract_class import NNPattern


class Classification(NNPattern):
    """
    Main class of the neural network to inherit.

    Methods:
            __init__(input_size, output_size)
                Initialize the class

            forward(x)
                forward() function of neural network

            bernoulliLayer()

            extract_data(df, aim_par, split_ratio)
                Prepare given data for training and validating the neural network

            train(lr, num_epochs)
                train() function of neural network

            predict()
                Validate (test) the neural network

            show_results()
                Demonstrate the loss/accuracy charts
    """

    # inputLikely = np.random(0, 1)
    # Applying the bernoulli class
    # data = bernoulli.rvs(size=1000, p=0.8)

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialize the class

        Arguments:
                input_size: int
                    The quantity of the input nodes (parameters)
                output_size: int
                    The quantity of the output nodes (parameters)

        Outputs:
                None
        """

        super(Classification, self).__init__(input_size, output_size)

        self.inputLayer = torch.nn.Linear(input_size, input_size).double()
        self.h1 = torch.nn.Tanh().double()
        self.h2 = torch.nn.Bilinear(5, 4, 9).double()
        self.h3 = torch.nn.Tanh().double()
        self.outputLayer = torch.nn.Linear(input_size, output_size).double()
        # self.linear = torch.nn.Linear(input_size, output_size).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward() function of neural network

        Arguments:
                x: torch.Tensor
                    Current state of weights

        Outputs:
                torch.Tensor:
                    Updated weights
        """

        # predict = self.linear(x)
        # return torch.sigmoid(predict)
        x = self.inputLayer(x)
        x = F.relu(self.h1(x))
        x = torch.sigmoid(self.h2(x[:5], x[5:]))
        x = F.relu(self.h3(x))
        x = self.outputLayer(x)
        return torch.sigmoid(x)

    def bernoulliLayer(self) -> None:
        pass

    def extract_data(self, df: pd.DataFrame, aim_par: str, split_ratio: float = .75) -> None:
        """
        Prepare given data for training and validating the neural network

        Arguments:
                df: pd.DataFrame
                    Input data
                aim_par: str
                    Goal parameter (label) that the neural network will predict
                split_ratio: float
                    Estimates the percentage in which train and test datasets will be separated

        Outputs:
                None
        """

        self.train_ds = []
        self.test_ds = []
        self.train_target = []
        self.test_target = []
        self.len_ds = df.shape[0]
        self.ds_sizes = {'train': int(self.len_ds * split_ratio),
                         'test': int(self.len_ds * (1. - split_ratio))}

        df = df.drop('Unnamed: 0', axis=1)
        self.train_target = np.array(df[aim_par][:self.ds_sizes['train']])
        self.test_target = np.array(df[aim_par][self.ds_sizes['train']:self.len_ds])
        df = df.drop(aim_par, axis=1)
        self.train_ds = np.array(df[:][:self.ds_sizes['train']])
        self.test_ds = np.array(df[:][self.ds_sizes['train']:self.len_ds])

        self.fig = None

    def draw_curve(self, cur_epoch: int) -> None:
        """
        Create and update loss and accuracy charts of the training phase

        Arguments:
                cur_epoch: int
                    Current epoch of training

        Outputs:
                None
        """

        # sns.set()
        self.x_epoch.append(cur_epoch)
        self.ax0.plot(self.x_epoch, self.y_loss, 'r-', label='train', linewidth=1)
        self.ax1.plot(self.x_epoch, self.y_err, 'g-', label='train', linewidth=1)
        if cur_epoch == 0:
            self.ax0.legend()
            self.ax1.legend()

    def train(self, lr: float = .0001, num_epochs: int = 50) -> None:
        """
        train() function of neural network

        Arguments:
                lr: float
                    The learning rate of the neural network
                num_epochs: int
                    The quantity of epochs to train the neural network

        Outputs:
                None
        """

        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.y_loss = []
        self.y_err = []
        self.x_epoch = []
        self.fig = plt.figure()
        self.ax0 = self.fig.add_subplot(121, title="Loss")
        self.ax1 = self.fig.add_subplot(122, title="Accuracy")

        for epoch in range(num_epochs):
            running_loss = 0.
            running_corrects = 0
            for i in range(self.ds_sizes['train']):
                if torch.cuda.is_available():
                    inputs = torch.tensor(self.train_ds[i], requires_grad=True, dtype=torch.float).cuda().double()
                    target = torch.tensor(self.train_target[i], requires_grad=True, dtype=torch.float).cuda().double()
                else:
                    inputs = torch.tensor(self.train_ds[i], requires_grad=True, dtype=torch.float).double()
                    target = torch.tensor(self.train_target[i], requires_grad=True, dtype=torch.float).double()

                optimizer.zero_grad()
                outputs = self.forward(inputs)
                target = target.unsqueeze(0)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_corrects += int(round(outputs.data.item()) == int(target.data.item()))

            epoch_loss = running_loss / self.ds_sizes['train']
            epoch_acc = running_corrects / self.ds_sizes['train']
            self.y_loss.append(epoch_loss)
            self.y_err.append(epoch_acc)
            self.draw_curve(epoch)

            print(f'Epoch {epoch + 1}/{num_epochs}:')
            print(f'Loss: {epoch_loss} Acc: {epoch_acc}\n\n')

    def predict(self) -> None:
        """
        Validate (test) the neural network

        Arguments:
                None

        Outputs:
                None
        """

        self.test_corrects = 0

        with torch.no_grad():
            for i in range(self.ds_sizes['test']):
                if torch.cuda.is_available():
                    inputs = torch.from_numpy(np.array(self.test_ds[i]).cuda()).double()
                    target = torch.from_numpy(np.array(self.test_target[i]).cuda()).double()
                else:
                    inputs = torch.from_numpy(np.array(self.test_ds[i])).double()
                    target = torch.from_numpy(np.array(self.test_target[i])).double()

                output = self.forward(inputs)
                self.test_corrects += int(round(output.data.item()) == round(target.data.item()))

    def show_results(self) -> None:
        """
        Demonstrate the loss/accuracy charts

        Arguments:
                None

        Outputs:
                None
        """

        if self.fig:
            self.fig.show()

        if self.test_corrects:
            colors = sns.color_palette('pastel')[:2]
            plt.pie([self.test_corrects, self.ds_sizes['test'] - self.test_corrects],
                    labels=['Correct', 'False'], colors=colors, autopct='%.2f%%',
                    explode=[0, .2], shadow=True)
            plt.show()
