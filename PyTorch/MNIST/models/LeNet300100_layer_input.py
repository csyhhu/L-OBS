import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet300100(nn.Module):

    def __init__(self):
        super(LeNet300100, self).__init__()
        self.layer_input = dict()

        self.fc1 = nn.Linear(784, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):

        x = x.view(-1, 784)
        self.layer_input['fc1'] = x.data
        x = F.relu(self.fc1(x))
        self.layer_input['fc2'] = x.data
        x = F.relu(self.fc2(x))
        self.layer_input['fc3'] = x.data
        x = self.fc3(x)
        return x

if __name__ == '__main__':

    net = LeNet300100()
    inputs = torch.rand([10,1,28,28])
    net(inputs)