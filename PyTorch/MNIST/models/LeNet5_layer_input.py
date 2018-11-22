import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.layer_input = dict()
        self.layer_kernel = {
            'conv1': 5, 'conv2': 5
        }
        self.layer_stride = {
            'conv1': 1, 'conv2': 1
        }

        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        self.layer_input['conv1'] = x.data
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        self.layer_input['conv2'] = x.data
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = x.view(-1, 800)
        self.layer_input['fc1'] = x.data
        x = F.relu(self.fc1(x))
        self.layer_input['fc2'] = x.data
        x = self.fc2(x)

        return x

if __name__ == '__main__':

    net = LeNet5()
    inputs = torch.rand([10,1,28,28])
    net(inputs)