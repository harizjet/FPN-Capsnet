import torch
import torch.nn.functional as F
from torch import nn

from lib.model.capsule import CapsuleLayer
from lib.model.ipyramid import ImagePyramid
from utils import batch_apply


class IPyramidCapsuleNet(nn.Module):
    """
    Example:
    $ model = IPyramidCapsuleNet()
    """
    def __init__(self, device=torch.device('cpu')):
        super(IPyramidCapsuleNet, self).__init__()

        self.device = device

        self.ipyramid = batch_apply.function(ImagePyramid(3, (28, 28)))
        
        # pipe 1 for p3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1)
        self.primary_capsules1 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=128, out_channels=32,
                                             kernel_size=3, stride=2, device=self.device)
        self.digit_capsules1 = CapsuleLayer(num_capsules=10, num_route_nodes=128, in_channels=8,
                                           out_channels=16, device=self.device)

        # pipe 2 for p2
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=1)
        self.primary_capsules2 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=128, out_channels=32,
                                             kernel_size=5, stride=2, device=self.device)
        self.digit_capsules2 = CapsuleLayer(num_capsules=10, num_route_nodes=288, in_channels=8,
                                           out_channels=16, device=self.device)

        # pipe 3 for p1
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=9, stride=1)
        self.primary_capsules3 = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=128, out_channels=32,
                                             kernel_size=9, stride=2, device=self.device)
        self.digit_capsules3 = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 6 * 6, in_channels=8,
                                           out_channels=16, device=self.device)

    def forward(self, x):
        x = torch.squeeze(x, 1)
        x1, x2, x3 = self.ipyramid(x)
        x1, x2, x3 = torch.unsqueeze(x1, 1), torch.unsqueeze(x2, 1), torch.unsqueeze(x3, 1)

        x1 = F.relu(self.conv1(x1), inplace=True)
        x1 = nn.BatchNorm2d(x1.size()[1]).to(self.device)(x1)
        x1 = self.primary_capsules1(x1)
        x1 = self.digit_capsules1(x1).squeeze(2).squeeze(2).transpose(0, 1)
        logits1 = (x1 ** 2).sum(dim=-1) ** 0.5

        x2 = F.relu(self.conv2(x2), inplace=True)
        x2 = nn.BatchNorm2d(x2.size()[1]).to(self.device)(x2)
        x2 = self.primary_capsules2(x2)
        x2 = self.digit_capsules2(x2).squeeze(2).squeeze(2).transpose(0, 1)
        logits2 = (x2 ** 2).sum(dim=-1) ** 0.5

        x3 = F.relu(self.conv3(x3), inplace=True)
        x3 = nn.BatchNorm2d(x3.size()[1]).to(self.device)(x3)
        x3 = self.primary_capsules3(x3)
        x3 = self.digit_capsules3(x3).squeeze(2).squeeze(2).transpose(0, 1)
        logits3 = (x3 ** 2).sum(dim=-1) ** 0.5

        logits = torch.stack((logits1, logits2, logits3), dim=1)
        logits = torch.sum(logits, 1)

        classes = F.softmax(logits, dim=-1)
        return classes