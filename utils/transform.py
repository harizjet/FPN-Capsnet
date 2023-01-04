import numpy as np

import torch


class augmentation(object):
    """
    Augment the dataset
    Example:
    $ t = torch.rand(1, 1, 28, 28)
    $ y = transform_augmentation((28, 28))(t)
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
        
    def __call__(self, sample):
        if self.isBatch:
            shifted_image = torch.zeros(*sample.shape)
            shifted_image[:, self.source_height_slice, self.source_width_slice] = sample[:, self.target_height_slice, self.target_width_slice]
        else:
            shifted_image = torch.zeros(*sample.shape)
            shifted_image[self.source_height_slice, self.source_width_slice] = sample[self.target_height_slice, self.target_width_slice]
        return shifted_image.float()


class toTensor(object):
    """
    Convert numpy array to Tensor
    Example:
    $ t = torch.rand(1, 1, 28, 28)
    $ y = transform_toTensor()(t)
    """

    def __call__(self, sample):
        return torch.from_numpy(np.asarray(sample))