import torch
from torch import nn
import torch.nn.functional as F


class CapsuleLoss(nn.Module):
    """
    Custom loss for capsule
    """
    def __init__(self):
        super(CapsuleLoss, self).__init__()
    
    def __repr__(self):
        return "CapsuleLoss"
    
    def forward(self, y, ypred):
        labels = torch.eye(10).to(ypred.device).index_select(dim=0, index=y)

        left = F.relu(0.9 - ypred, inplace=True) ** 2
        right = F.relu(ypred - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        return margin_loss


class CrossEntropyLoss(nn.Module):
    """
    Cross Entropy Loss
    """

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.critique = nn.CrossEntropyLoss()
    
    def __repr__(self):
        return "CrossEntropyLoss"
    
    def forward(self, y, ypred):
        ypred = torch.max(ypred, dim=-1).values

        return self.critique(y.float(), ypred)
        