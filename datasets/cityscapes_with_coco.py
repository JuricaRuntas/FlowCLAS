import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import Cityscapes
from yacs.config import CfgNode
from rich.progress import Progress
from datasets.coco_dataset import CocoDatasetWithoutCityscapesClasses
from PIL import Image
from typing import List
from datasets.transforms import DINOv2_transforms

def _get_target_suffix_train(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            #return f"{mode}_labelIds.png"
            return f"{mode}_labelTrainIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"

# TODO: write this in a more civilized way
Cityscapes._get_target_suffix = _get_target_suffix_train
        
class CityscapesWithCocoDataset(Dataset):
    """Auxiliary dataset that combines Cityscapes with COCO."""
    def __init__(self, cfg: CfgNode):        
        self.cityscapes_dataset = Cityscapes(
            root=cfg.DATASETS.CITYSCAPES.ROOT,
            split=cfg.DATASETS.CITYSCAPES.SPLIT,
            mode=cfg.DATASETS.CITYSCAPES.MODE,
            target_type=cfg.DATASETS.CITYSCAPES.TARGET_TYPE
        )
        
        self.coco_dataset = CocoDatasetWithoutCityscapesClasses(
            root=cfg.DATASETS.COCO.ROOT,
            json_annotation_file=cfg.DATASETS.COCO.ANNOTATIONS,
            min_size=cfg.DATASETS.COCO.MIN_SIZE
        )
        
        if "dinov2" in cfg.BACKBONE.ARCHITECTURE:
            self.transform, self.target_transform = DINOv2_transforms(cfg)
        else:
            self.transform, self.target_transform = None, None
        
        self.subsampling_factor = cfg.BACKBONE.PATCH_SIZE
        self.ignore_label = cfg.DATASETS.IGNORE_LABEL
        
        self.random_rescale_lower = cfg.DATASETS.COCO.RANDOM_RESCALE_LOWER
        self.random_rescale_upper = cfg.DATASETS.COCO.RANDOM_RESCALE_UPPER
                
        # Randomly select COCO images
        self.selected_coco_ids = np.random.choice(range(len(self.coco_dataset)), len(self.cityscapes_dataset), replace=False)

    def paste_anomalous_instance(self, cityscapes_image: np.ndarray, cityscapes_target: np.ndarray, 
                                 coco_image: np.ndarray, coco_instance_mask: np.ndarray):
        """Paste an anomalous instance from COCO image onto a Cityscapes image."""
        ood_h, ood_w = coco_image.shape[:2]
        id_h, id_w = cityscapes_image.shape[:2]
        
        loc_x = np.random.randint(0, id_w - ood_w)
        loc_y = np.random.randint(0, id_h - ood_h)
        
        for channel in range(3):
            cityscapes_image[loc_y:loc_y+ood_h, loc_x:loc_x+ood_w, channel] = (
                cityscapes_image[loc_y:loc_y+ood_h, loc_x:loc_x+ood_w, channel] * (1 - coco_instance_mask) +
                coco_instance_mask * coco_image[:, :, channel]
            )
            
        cityscapes_target[loc_y:loc_y+ood_h, loc_x:loc_x+ood_w][coco_instance_mask == 1] = 1
        
        return Image.fromarray(cityscapes_image), Image.fromarray(cityscapes_target)
            
    def create_mixed_image(self, cityscapes_image: Image.Image, cityscapes_target: Image.Image, 
                           coco_image: Image.Image, coco_annotations: List[dict]):
        """Creates a mixed image by pasting an anomalous instance from COCO image onto Cityscapes image."""
        
        cityscapes_image_np = np.array(cityscapes_image)
        cityscapes_target_np = np.array(cityscapes_target)
                
        num_instances = 1
        
        if len(coco_annotations) >= 3:
            num_instances = 3
        elif len(coco_annotations) == 2:
            num_instances = 2
        
        selected_annotations = np.random.choice(coco_annotations, size=num_instances, replace=False)
        
        for coco_annotation in selected_annotations:
            coco_instance_mask = self.coco_dataset.coco.annToMask(coco_annotation)
            
            if coco_instance_mask.sum() < 100:  # Minimum 100 pixels
                continue

            size = np.random.randint(self.random_rescale_lower, self.random_rescale_upper + 1)
            factor = size / max(coco_instance_mask.shape)
            
            coco_image_tensor = torch.from_numpy(np.array(coco_image)).float().permute(2, 0, 1).unsqueeze(0)
            coco_mask_tensor = torch.from_numpy(coco_instance_mask).unsqueeze(0).unsqueeze(0).float()
            
            resized_image = F.interpolate(coco_image_tensor, scale_factor=factor, mode="bilinear", align_corners=False)
            resized_mask = F.interpolate(coco_mask_tensor, scale_factor=factor, mode="nearest")

            resized_image_np = resized_image.squeeze(0).permute(1, 2, 0).clamp(0, 255).byte().numpy()
            resized_mask_np = resized_mask.squeeze(0).squeeze(0).byte().numpy()
                        
            cityscapes_image_pil, cityscapes_target_pil = self.paste_anomalous_instance(
                cityscapes_image=cityscapes_image_np,
                cityscapes_target=cityscapes_target_np,
                coco_image=resized_image_np,
                coco_instance_mask=resized_mask_np
            )
        
            cityscapes_image_np = np.array(cityscapes_image_pil)
            cityscapes_target_np = np.array(cityscapes_target_pil)
        
        return Image.fromarray(cityscapes_image_np), Image.fromarray(cityscapes_target_np)
        
            
    def __getitem__(self, index):
        cityscapes_image, cityscapes_target = self.cityscapes_dataset[index]
        
        # The pasted anomalous object will be labeled as 1 resulting in a 
        # binary segmentation mask for the mixed image, however, we keep ignore label for later processing
        cityscapes_target = np.array(cityscapes_target)
        cityscapes_target[cityscapes_target != self.ignore_label] = 0
        cityscapes_target = Image.fromarray(cityscapes_target)
        
        coco_index = np.random.randint(len(self.selected_coco_ids))
        coco_image, coco_annotations = self.coco_dataset[int(self.selected_coco_ids[coco_index])]
        
        mixed_image, mixed_target = self.create_mixed_image(cityscapes_image, cityscapes_target, coco_image, coco_annotations)
        
        if self.transform is not None:
            coco_image = self.transform(coco_image)
            mixed_image = self.transform(mixed_image)
        
        if self.target_transform is not None:
            mixed_target = self.target_transform(mixed_target)
            mixed_target = F.interpolate(mixed_target.unsqueeze(0).float(), 
                                         size=(mixed_image.shape[-2]//self.subsampling_factor, mixed_image.shape[-1]//self.subsampling_factor), 
                                         mode="nearest")[0, 0].long()   
            
        return coco_image, mixed_image, mixed_target        
        
    def __len__(self):
        return len(self.cityscapes_dataset)
    
    @property
    def name(self):
        return "CityscapesWithCoco"
