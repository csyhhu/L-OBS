import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):

        x = self.conv1(x)
        # print(x.shape)
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(F.max_pool2d(x, kernel_size=2, stride=2))
        # print(x.shape)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        x = self.fc2(x)
        # print(x.shape)
        return x

if __name__ == '__main__':

    net = LeNet5()
    inputs = torch.rand([10,1,28,28])
    net(inputs)