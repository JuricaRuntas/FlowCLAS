import glob
import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from datasets.transforms import DINOv2_transforms
from yacs.config import CfgNode

class RoadDrivingOODDataset(Dataset):
    def __init__(self, cfg: CfgNode, root: Path, transform=None, target_transform=None):
        """
        Args:
            cfg (CfgNode): Configuration node containing dataset parameters.
            root (Path): Path to the directory containing images.
            transform: A function/transform that takes in an PIL image and returns a transformed version.
            target_transform: A function/transform that takes in the target and transforms it.
        """
        self.cfg = cfg
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
                
        if root == Path(cfg.DATASETS.ROAD_ANOMALY.ROOT):
            self.images = list(sorted(glob.glob(f"{root}/frames/*.jpg")))
            self.targets = [im_file.replace(".jpg", ".labels/labels_semantic.png") for im_file in self.images]
        
        elif root == Path(cfg.DATASETS.FISHYSCAPES_LAF.ROOT):
            self.images = list(sorted(glob.glob(f"{root}/*.png")))
            self.targets = list(sorted(glob.glob(f"{root}/labels/*.png")))
        
        elif root == Path(cfg.DATASETS.FISHYSCAPES_STATIC.ROOT):
            self.images = list(sorted(glob.glob(f"{root}/*.jpg")))
            self.targets = [im_file.replace("_rgb.jpg", "_labels.png") for im_file in self.images]
            
        else:
            raise NotImplementedError(f"RoadDrivingOODDataset at {root} is not supported.")
        
        assert len(self.images) == len(self.targets), "Number of images and labels does not match."
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")        
        target = np.array(Image.open(self.targets[index]).convert("L"))
        
        if self.root == self.cfg.DATASETS.ROAD_ANOMALY.ROOT:
            target[target == 2] = 1
            
        target = Image.fromarray(target)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.images[index], self.targets[index]
    
    
class RoadAnomaly(RoadDrivingOODDataset):
    def __init__(self, cfg: CfgNode):
        root = Path(cfg.DATASETS.ROAD_ANOMALY.ROOT)
        if "dinov2" in cfg.BACKBONE.ARCHITECTURE:
            transform, target_transform = DINOv2_transforms(cfg)
        super().__init__(cfg, root, transform=transform, target_transform=target_transform)
    
    @property
    def name(self):
        return "RoadAnomaly"
        
class FishyscapesLostAndFound(RoadDrivingOODDataset):
    def __init__(self, cfg: CfgNode):
        root = Path(cfg.DATASETS.FISHYSCAPES_LAF.ROOT)
        if "dinov2" in cfg.BACKBONE.ARCHITECTURE:
            transform, target_transform = DINOv2_transforms(cfg)
        super().__init__(cfg, root, transform=transform, target_transform=target_transform)
    
    @property
    def name(self):
        return "FishyscapesLostAndFound"    
    
class FishyscapesStatic(RoadDrivingOODDataset):
    def __init__(self, cfg: CfgNode):
        root = Path(cfg.DATASETS.FISHYSCAPES_STATIC.ROOT)
        if "dinov2" in cfg.BACKBONE.ARCHITECTURE:
            transform, target_transform = DINOv2_transforms(cfg)
        super().__init__(cfg, root, transform=transform, target_transform=target_transform)
        
    @property
    def name(self):
        return "FishyscapesStatic"