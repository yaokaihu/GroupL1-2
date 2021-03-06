import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv2d, Linear, BatchNorm2d


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=200):
        super(resNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))).cuda()
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1).cuda()
        out = self.linear(out).cuda()
        return out


class resnet20(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(resnet20, self).__init__()
        self.in_planes = 16

        self.conv1 = Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 6)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return resNet18(BasicBlock, [2, 2, 2, 2])


def ResNet20(num_classes=10):
    return resnet20(BasicBlock, [3, 3, 3], num_classes)
