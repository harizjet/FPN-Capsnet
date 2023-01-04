import cv2
import torch
import numpy as np


class ImagePyramid(object):
    """
    Extract Image pyramid using Gaussian and Laplacian
    """
    def __init__(self, num_times: int, origin_shape: tuple[int, int]):
        self.num_times = num_times
        self.origin_shape = origin_shape
    
    def __call__(self, sample) -> list:
        x, y = sample.shape
        assert (x, y) == self.origin_shape
        
        has_device = False

        if not isinstance(sample, np.ndarray):
            device = sample.device
            if device.type != 'cpu':
                has_device = True
                sample = sample.cpu()

            sample = sample.numpy()
            
        output = []
        based = sample
        for i in range(self.num_times-1):
            based = cv2.pyrDown(based)
        for i in range(self.num_times):
            if i != 0:
                based = cv2.pyrUp(based)
            val = torch.from_numpy(based)
            if has_device:
                val = val.to(device)
            output.append(val)

        return output