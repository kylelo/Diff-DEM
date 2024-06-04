import torch
import torch.nn as nn

class DEMNormalize(object):
    def __call__(self, height_meter):
        min_val, max_val = torch.min(height_meter), torch.max(height_meter)
        diff = max_val - min_val
        if diff == 0:
            return torch.zeros(height_meter.shape)
        height = (height_meter - min_val) / diff
        height = 2 * height - 1
        height = torch.clip(height, -1, 1)
        return height
    
class MaxPooling2DTransform(object):
    def __init__(self, kernel_size, stride):
        self.max_pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0)

    def __call__(self, sample):
        # Assuming sample is already a tensor of shape [C, H, W]
        # Apply max pooling
        return self.max_pool(sample)

class ToFloat32(object):
    def __call__(self, tensor):
        return tensor.to(torch.float32)