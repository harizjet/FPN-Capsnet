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
        
        if not isinstance(sample, np.ndarray):
            sample = sample.numpy()
            
        output = []
        based = sample
        for i in range(self.num_times-1):
            based = cv2.pyrDown(based)
        for i in range(self.num_times):
            if i != 0:
                based = cv2.pyrUp(based)
            output.append(torch.from_numpy(based))
        return output