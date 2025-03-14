import torch.nn.functional as F
import torch.nn as nn
class BasicBlock3(nn.Module):
    expansion = 1


    def __init__(self, in_planes, planes, stride=1, kernel=3):
        super(BasicBlock3, self).__init__()

        DROPOUT = 0.1
        self.kernel = kernel
        padding = kernel // 2
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=self.kernel, stride=stride, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=self.kernel, stride=1, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=self.kernel,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)


        self.bn3 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = F.relu(self.dropout(self.bn2(self.conv2(out))))
        out = self.dropout(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Res3NetK533(nn.Module):
    def __init__(self, block, num_blocks=[2, 2, 2], num_classes=10):
        super(Res3NetK533, self).__init__()
        self.in_planes = 64


        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, kernel=5)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear1 = nn.Linear(256*block.expansion, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, kernel=3):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, kernel))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = F.relu(out)
        out = self.linear2(out)
        return F.log_softmax(out, dim=-1)