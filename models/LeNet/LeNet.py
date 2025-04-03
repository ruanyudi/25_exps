import torch.nn as nn
import torch


class LeNet(nn.Module):
    def __init__(self, Numclass):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*32*32)
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv2 = nn.Sequential(nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1 = nn.Sequential(nn.Linear(16 * 5 * 5, 120), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(120, 84), nn.ReLU())
        self.fc3 = nn.Linear(84, 10)

    #  forward definition where x is the input
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        fc = self.fc2(x)
        x = self.fc3(fc)
        # x = F.log_softmax(x, dim=1)
        return x
