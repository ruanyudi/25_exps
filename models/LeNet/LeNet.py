import torch.nn as nn
import torch


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*32*32)
            nn.Conv2d(
                1, 6, 5
            ),  # in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0
            # 28*28*6
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 14*14*6
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                6, 16, 5
            ),  # in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0
            # 10*10*16
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # 5*5*16
        )

    #  forward definition where x is the input
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size()[0], -1)  # 400
        return x
