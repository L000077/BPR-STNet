import torch
from torch import nn
from torchsummary import summary

# Depthwise Separable Convolutional Layer
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
        super(SeparableConv2d, self).__init__()

        # groups=in_channels=out_channels
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                                        padding=1, groups=in_channels, bias=bias)
        # 1*1 kernel size
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # print(x.dtype)
        x = x.to(torch.float32)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x

# Residual Block
class residual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(residual, self).__init__()

        self.sepConv1 = SeparableConv2d(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.sepConv2 = SeparableConv2d(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        # skip connection
        if out_channels != in_channels or stride != 1:
            # self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=True)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.sepConv1(x)
        y = self.bn1(y)
        y = self.relu1(y)
        y = self.sepConv2(y)
        y = self.bn2(y)
        if self.conv3:
            x = x.to(torch.float32)
            x = self.conv3(x)
        y = x + y
        y = self.relu2(y)
        return y

# BPR-STNet
class net(nn.Module):
    def __init__(self, num_classes=2):
        super(net, self).__init__()
        self.num_class = num_classes

        self.pw = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            residual(32, 32, 1),
            residual(32, 32, 1)
        )
        self.layer2 = nn.Sequential(
            residual(32, 64, 2),
            residual(64, 64, 1)
        )

        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.pw(x)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        output = self.softmax(x)

        return output
