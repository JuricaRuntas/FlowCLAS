import torch
import torchvision
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Compose, ToTensor, Normalize, PILToTensor
from yacs.config import CfgNode

# adapted from https://github.com/vojirt/PixOOD/blob/main/code/dataloaders/augmentations.py
class ResizeLongestSideDivisible:
    def __init__(self, divider: int):
        """
        Resize the longest side of the image to be divisible by given divider.
        
        Args:
            divider (int): The value by which the longest side of the image should be divisible.
        """
        self.divider = divider

    def __call__(self, x):
        x_size = x.shape[-2:]

        self.img_sz = int((max(x_size) // self.divider) * self.divider)

        if x_size[0] >= x_size[1]:
            factor = x_size[0] / float(self.img_sz)
            size = [int(self.img_sz), int(self.divider*((x_size[1] / factor) // self.divider))] 
        else:
            factor = x_size[1] / float(self.img_sz)
            size = [int(self.divider*((x_size[0] / factor) // self.divider)), int(self.img_sz)] 

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

    # TODO: decide whether to use ResizeLongestSideDivisible for labels
    #labels_transform = Compose([PILToTensor(), ResizeLongestSideDivisible(cfg.BACKBONE.PATCH_SIZE)])
    labels_transform = Compose([PILToTensor()])

    return transform, labels_transform