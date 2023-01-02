import numpy as np
import matplotlib.pyplot as plt
import torch
import torchnet as tnt
from torchvision.datasets.mnist import MNIST


def augmentation(x, not_batch=True, max_shift=2):
    """
    Example:
    $ t = torch.rand(1, 1, 28, 28)
    $ y = augmentation(t)
    """
    if not_batch:
        height, width = x.size()
    else:
        _, _, height, width = x.size()
        
    h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    source_height_slice = slice(max(0, h_shift), h_shift + height)
    source_width_slice = slice(max(0, w_shift), w_shift + width)
    target_height_slice = slice(max(0, -h_shift), -h_shift + height)
    target_width_slice = slice(max(0, -w_shift), -w_shift + width)
    shifted_image = torch.zeros(*x.size())

    if not_batch:
        shifted_image[source_height_slice, source_width_slice] = x[target_height_slice, target_width_slice]
    else:
        shifted_image[:, :, source_height_slice, source_width_slice] = x[:, :, target_height_slice, target_width_slice]
    return shifted_image.float()
