import numpy as np

import torch
import cv2
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import torchvision.transforms.functional as Fn
import torchvision.transforms as T


class augmentation(object):
    """
    Augment the dataset
    Example:
    $ t = torch.rand(1, 1, 28, 28)
    $ y = augmentation((28, 28), isBatch=True)(t)
    """
    def __init__(self, image_size, max_shift=2, isBatch=False):
        assert isinstance(image_size, tuple)
        height, width = image_size
        self.image_size = image_size
        self.isBatch = isBatch

        h_shift, w_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
        self.source_height_slice = slice(max(0, h_shift), h_shift + height)
        self.source_width_slice = slice(max(0, w_shift), w_shift + width)
        self.target_height_slice = slice(max(0, -h_shift), -h_shift + height)
        self.target_width_slice = slice(max(0, -w_shift), -w_shift + width)
        
    def _shift(self, sample):
        nonzero_x_cols = (torch.sum(torch.from_numpy(sample.numpy()), dim=0) > 0).nonzero()
        nonzero_y_cols = (torch.sum(torch.from_numpy(sample.numpy()), dim=1) > 0).nonzero()
        left_margin = torch.min(nonzero_x_cols)
        right_margin = 28 - torch.max(nonzero_x_cols) - 1
        top_margin = torch.min(nonzero_y_cols)
        bot_margin = 28 - torch.max(nonzero_y_cols) - 1
        rand_dirs = torch.rand([2])
        dir_idxs = torch.floor(rand_dirs * 2).long()
        rand_amts = torch.min(torch.abs(torch.randn([2]) * 0.33), torch.tensor(0.999))
        x_amts = [torch.floor(-1 * rand_amts[0] * left_margin), torch.floor(rand_amts[0] * (1 + right_margin))]
        y_amts = [torch.floor(-1 * rand_amts[1] * top_margin), torch.floor(rand_amts[1] * (1 + bot_margin))]
        x_amt = torch.gather(torch.tensor(x_amts), index=dir_idxs[1], dim=0)
        y_amt = torch.gather(torch.tensor(y_amts), index=dir_idxs[0], dim=0)
        shifted_image = sample.reshape(28 * 28)
        shifted_image = torch.roll(shifted_image, (y_amt*28).long().item(), dims=0)
        shifted_image = shifted_image.reshape(28, 28)
        shifted_image = shifted_image.transpose(0, 1)
        shifted_image = shifted_image.reshape(28 * 28)
        shifted_image = torch.roll(shifted_image, (x_amt*28).long().item(), dims=0)
        shifted_image = shifted_image.reshape(28, 28)
        shifted_image = shifted_image.transpose(0, 1)
        return shifted_image.float()
    
    def _rotate(self, sample):
        rand_amts = torch.max(torch.min(torch.randn(2) * 0.33, torch.tensor(0.9999)), torch.tensor(-0.9999))
        angle = rand_amts[0] * 30
        rot_mat = cv2.getRotationMatrix2D((28/2, 28/2), int(angle), 1.0)
        rotated = torch.from_numpy(cv2.warpAffine(sample.numpy(), rot_mat, (28, 28)))
        rotated_image = sample if rand_amts[1] > 0 else rotated
        return rotated_image.float()

    def _squish(self, sample):
        rand_amts = torch.min(torch.abs(torch.randn(2) * 0.33), torch.tensor(0.999))
        width_mod = torch.floor((rand_amts[0] * (28 / 4)) + 1).long()
        offset_mod = torch.floor(rand_amts[1] * 2).long()
        offset = ((width_mod // 2) + offset_mod).long()
        squished_image = Fn.resize(T.ToPILImage()(sample), [28, 28-width_mod], interpolation=InterpolationMode.LANCZOS)
        squished_image = F.pad(Fn.pil_to_tensor(squished_image)[0], (offset, offset_mod+width_mod-offset, 0, 0))
        squished_image = Fn.crop(squished_image, 0, 0, 28, 28)
        return squished_image.float()

    def _erase(self, sample):
        rand_amts = torch.rand(2)
        x = torch.floor(rand_amts[0]*19).long()+4
        y = torch.floor(rand_amts[1]*19).long()+4
        patch = torch.zeros([4, 4])
        mask = F.pad(patch, (x, 28-x-4, y, 28-y-4), value=1)
        patch_image = mask * sample
        return patch_image.float()

    def _individual(self, sample):
        sample = self._rotate(sample)
        sample = self._shift(sample)
        sample = self._squish(sample)
        sample = self._erase(sample)
        return sample.float()
        
    def __call__(self, sample):
        if not self.isBatch:
            return self._individual(sample)
    
        return torch.stack(list(map(self._individual, sample)))


class toTensor(object):
    """
    Convert numpy array to Tensor
    Example:
    $ t = torch.rand(1, 1, 28, 28)
    $ y = toTensor()(t)
    """

    def __call__(self, sample):
        return torch.from_numpy(np.asarray(sample)).type(torch.uint8)


class scale(object):
    """
    Scale images
    Example:
    $ t = torch.rand(1, 1, 28, 28)
    $ y = scale()(t)
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        return (sample-self.mean)/self.std