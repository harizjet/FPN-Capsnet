import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from lib.model.capsule import CapsuleLayer


class CapsuleNet(nn.Module):
    """
    Example:
    $ model = CapsuleNet()
    """
    def __init__(self, device=torch.device('cpu')):
        super(CapsuleNet, self).__init__()

        self.device = device

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                             kernel_size=9, stride=2, device=self.device)
        self.digit_capsules = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16, device=self.device)

    def forward(self, x, y=None):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.primary_capsules(x)
        x = self.digit_capsules(x).squeeze(2).squeeze(2).transpose(0, 1)
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        return classes
