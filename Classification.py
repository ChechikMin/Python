import torch.nn
import numpy as np
import torch.nn.functional as F
from scipy.stats import bernoulli

class Classification(torch.nn.Module):
    inputLikely = np.random(0, 1)
    # Applying the bernoulli class
    data = bernoulli.rvs(size=1000, p=0.8)

    def __init__(self, inputSize, out = 1):
        super(Classification, self).__init__()
        self.inputLayer = torch.nn.Linear(inputSize, inputSize).double()
        self.fc2 = torch.nn.Linear(inputSize, inputSize).double()
        self.fc3 = torch.nn.Tanh(inputSize, inputSize).double()
        self.fc4 = torch.nn.Bilinear(inputSize, inputSize).double()
        self.outpuLayer = torch.nn.Linear(inputSize, out).double()

    def bernoulliLayer(self):
        pass
    def train(self) -> float:
        return 1
    def predict(self) -> bool:
        return 1
    def showResults(self):
        pass

    def forward(self, x):
        x = self.inputLayer(x)
        x = F.relu(self.fc2(x))
        x = F.sigmoid(self.fc4)
        x = self.outpuLayer(x)
        return F.log_softmax(x)