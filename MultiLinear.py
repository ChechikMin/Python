from tkinter import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn
from torch.autograd import Variable


class linearRegression(torch.nn.Module):
    def __init__(self, sizeInTrain, sizeOutTrain):
        super(linearRegression,self).__init__()
        self.linear = torch.nn.Linear(sizeInTrain, sizeOutTrain).double()

    def forward(self, train_x):
        return self.linear(train_x)
