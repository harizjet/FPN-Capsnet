import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from utils.capsule import CapsuleLayer
from utils.ipyramid import ImagePyramid


class IPyramidCapsuleNet(nn.Module):
    """
    Example:
    $ model = IPyramidCapsuleNet()
    """
    def __init__(self, use_cuda=False):
        super(IPyramidCapsuleNet, self).__init__()

        self.use_cuda = use_cuda

        self.ipyramid = ImagePyramid(3, (28, 28))
        
        # pipe 1 for p3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules1 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules1 = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        # pipe 2 for p2
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules2 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules2 = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

        # pipe 3 for p1
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules3 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2)
        self.digit_capsules3 = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16)

    def forward(self, x, y=None):
        x1 = F.relu(self.conv1(x), inplace=True)
        x1 = self.primary_capsules1(x1)
        x1 = self.digit_capsules1(x1).squeeze(2).squeeze(2).transpose(0, 1)
        logits1 = (x1 ** 2).sum(dim=-1) ** 0.5

        x2 = F.relu(self.conv2(x), inplace=True)
        x2 = self.primary_capsules1(x2)
        x2 = self.digit_capsules1(x2).squeeze(2).squeeze(2).transpose(0, 1)
        logits2 = (x2 ** 2).sum(dim=-1) ** 0.5

        x3 = F.relu(self.conv3(x), inplace=True)
        x3 = self.primary_capsules1(x3)
        x3 = self.digit_capsules1(x3).squeeze(2).squeeze(2).transpose(0, 1)
        logits3 = (x3 ** 2).sum(dim=-1) ** 0.5

        logits = torch.vstack((logits1, logits2, logits3))
        logits = torch.sum(logits, 0)

        classes = F.softmax(logits, dim=-1)

        return classes
