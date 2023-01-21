import torch
import torch.nn.functional as F
from torch import nn

from lib.model.capsule import CapsuleLayer


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FPNCapsuleNet(nn.Module):
    """
    Example:
    $ model = FPNCapsuleNet()
    """
    def __init__(self, block, num_blocks, device=torch.device('cpu')):
        super(FPNCapsuleNet, self).__init__()

        self.device = device

        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Bottom-up layers
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=1)

        # Top layer
        self.toplayer = nn.Conv2d(256, 32, kernel_size=1, stride=1, padding=0)  # Reduce channels

        # Smooth layers
        self.smooth1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        # Lateral layers
        self.latlayer1 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)

        # Capsule for pipe1
        self.primary_capsules1 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=32, out_channels=32,
                                             kernel_size=3, stride=2, device=self.device)
        self.digit_capsules1 = CapsuleLayer(num_capsules=10, num_route_nodes=512, in_channels=8,
                                           out_channels=16, device=self.device)

        # Capsule for pipe2
        self.primary_capsules2 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=32, out_channels=32,
                                             kernel_size=5, stride=2, device=self.device)
        self.digit_capsules2 = CapsuleLayer(num_capsules=10, num_route_nodes=288, in_channels=8,
                                           out_channels=16, device=self.device)

        # Capsule for pipe3
        self.primary_capsules3 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=32, out_channels=32,
                                             kernel_size=9, stride=2, device=self.device)
        self.digit_capsules3 = CapsuleLayer(num_capsules=10, num_route_nodes=32, in_channels=8,
                                           out_channels=16, device=self.device)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear') + y

    def _get_feature(self, x):
        # Bottom-up
        c1 = F.relu(self.conv1(x))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        # Top-down
        p3 = self.toplayer(c3)
        p2 = self._upsample_add(p3, self.latlayer1(c2))
        p1 = self._upsample_add(p2, self.latlayer2(c1))
        # Smooth
        p2 = self.smooth1(p2)
        p1 = self.smooth2(p1)
        
        return p1, p2, p3

    def forward(self, x):
        p1, p2, p3 = self._get_feature(x)

        p1 = nn.BatchNorm2d(p1.size()[1]).to(self.device)(p1)
        p2 = nn.BatchNorm2d(p2.size()[1]).to(self.device)(p2)
        p3 = nn.BatchNorm2d(p3.size()[1]).to(self.device)(p3)

        x1 = F.relu(p1, inplace=True)
        x1 = self.primary_capsules1(x1)
        x1 = self.digit_capsules1(x1).squeeze(2).squeeze(2).transpose(0, 1)
        logits1 = (x1 ** 2).sum(dim=-1) ** 0.5
        
        x2 = F.relu(p2, inplace=True)
        x2 = self.primary_capsules2(x2)
        x2 = self.digit_capsules2(x2).squeeze(2).squeeze(2).transpose(0, 1)
        logits2 = (x2 ** 2).sum(dim=-1) ** 0.5

        x3 = F.relu(p3, inplace=True)
        x3 = self.primary_capsules3(x3)
        x3 = self.digit_capsules3(x3).squeeze(2).squeeze(2).transpose(0, 1)
        logits3 = (x3 ** 2).sum(dim=-1) ** 0.5
        logits = torch.stack((logits1, logits2, logits3), dim=1)
        logits = torch.sum(logits, 1)

        classes = F.softmax(logits, dim=-1)
        return classes


def FPN101CapsuleNet(device=torch.device('cpu')):
    """
    Example:
    $ net = FPN101CapsuleNet()
    $ output = net(Variable(torch.randn(1, 1, 28, 28)))
    """
    return FPNCapsuleNet(Bottleneck, [2,2], device=device)