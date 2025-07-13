import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, PILToTensor
from yacs.config import CfgNode

# adapted from https://github.com/vojirt/PixOOD/blob/main/code/dataloaders/transforms.py
class ResizeLongestSideDivisible:
    def __init__(self, divider: int):
        """
        Resize the input image so that its longest side becomes the largest multiple of `divider`
        less than or equal to the original size, and the other side is scaled to preserve aspect ratio,
        rounded down to the nearest multiple of `divider`.

        This ensures both sides of the output are divisible by `divider`, which is useful for models
        that require input dimensions to be multiples of a specific value.

        Args:
            divider (int): The value by which both sides of the output image should be divisible.
        """
        self.divider = divider

    def __call__(self, x):
        x_size = x.shape[-2:]

        img_size = int((max(x_size) // self.divider) * self.divider)

        if x_size[0] >= x_size[1]:
            factor = x_size[0] / float(img_size)
            size = [int(img_size), int(self.divider*((x_size[1] / factor) // self.divider))] 
        else:
            factor = x_size[1] / float(img_size)
            size = [int(self.divider*((x_size[0] / factor) // self.divider)), int(img_size)] 

        if x.dtype == torch.float32:
            return torchvision.transforms.functional.resize(x, size, antialias=True)
        elif x.dtype == torch.uint8:
            return torchvision.transforms.functional.resize(x[None, ...], size, interpolation=InterpolationMode.NEAREST)[0, ...]
        else:
            raise NotImplementedError


def DINOv2_transforms(cfg: CfgNode):
    transform = Compose([ToTensor(), 
                         ResizeLongestSideDivisible(cfg.BACKBONE.PATCH_SIZE), 
                         Normalize(mean=cfg.BACKBONE.NORM_MEAN, std=cfg.BACKBONE.NORM_STD)])
    labels_transform = Compose([PILToTensor()])
    return transform, labels_transform


def pad_to_shape(x, target_shape):
    """
    Pads the input tensor `x` to match the target shape.
    
    Args:
        x (torch.Tensor): Input tensor of shape (C, H, W).
        target_shape (tuple): Target shape (H, W) to pad to.
        
    Returns:
        torch.Tensor: Padded tensor of shape (C, target_shape[0], target_shape[1]).
    """
    C, H, W = x.shape
    target_H, target_W = target_shape
    
    if H < target_H or W < target_W:
        padding = (0, max(0, target_W - W), 0, max(0, target_H - H))
        return F.pad(x, padding)
    
    return x
