import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset, Cityscapes
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
        
class CityscapesWithCocoDataset(VisionDataset):
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
            min_size=cfg.DATASETS.COCO.MIN_SIZE,
            ood_id=cfg.DATASETS.COCO.OOD_ID,
            id_id=cfg.DATASETS.COCO.ID_ID
        )
        
        if "dinov2" in cfg.BACKBONE.ARCHITECTURE:
            self.transform, self.target_transform = DINOv2_transforms(cfg)
        else:
            self.transform, self.target_transform = None, None
        
        self.ood_id = cfg.DATASETS.COCO.OOD_ID
        self.id_id = cfg.DATASETS.COCO.ID_ID
        
        self.random_rescale_lower = cfg.DATASETS.COCO.RANDOM_RESCALE_LOWER
        self.random_rescale_upper = cfg.DATASETS.COCO.RANDOM_RESCALE_UPPER
        
        self.max_OOD_instances_to_paste = cfg.DATASETS.COCO.MAX_OOD_INSTANCES_TO_PASTE
        
        # Randomly select COCO images to match the number of Cityscapes images
        self.selected_coco_ids = np.random.choice(range(len(self.coco_dataset)), len(self.cityscapes_dataset), replace=False)


    def paste_anomalous_instance(self, cityscapes_image: np.ndarray, cityscapes_target: np.ndarray, 
                               coco_image: np.ndarray, coco_instance_mask: np.ndarray, ood_id=254, id_id=0):
        """Pastes an anomalous instance from COCO image onto a Cityscapes image."""
        ood_h, ood_w = coco_image.shape[:2]
        id_h, id_w = cityscapes_image.shape[:2]
        
        loc_x = np.random.randint(0, id_w - ood_w)
        loc_y = np.random.randint(0, id_h - ood_h)
        
        for channel in range(3):
            cityscapes_image[loc_y:loc_y+ood_h, loc_x:loc_x+ood_w, channel] = (
                cityscapes_image[loc_y:loc_y+ood_h, loc_x:loc_x+ood_w, channel] * (1 - coco_instance_mask) +
                coco_instance_mask * coco_image[:, :, channel]
            )
            
        cityscapes_target[loc_y:loc_y+ood_h, loc_x:loc_x+ood_w][coco_instance_mask == 1] = ood_id
        
        return cityscapes_image, cityscapes_target
            

    def create_mixed_image(self, cityscapes_image: Image.Image, cityscapes_target: Image.Image, 
                      coco_image: Image.Image, coco_target: List[dict], ood_id=254, id_id=0):
        """Creates a mixed content image by pasting anomalous instance from COCO onto Cityscapes images."""
        
        coco_instance_mask = self.coco_dataset.coco.annToMask(np.random.choice(coco_target))
        
        size = np.random.randint(self.random_rescale_lower, self.random_rescale_upper)
        factor = size / max(coco_instance_mask.shape)
        
        if factor < 1.0:
            coco_image = torch.nn.functional.interpolate(
                torch.from_numpy(np.array(coco_image)).float().permute(2, 0, 1).unsqueeze(0), 
                scale_factor=factor)[0].permute(1, 2, 0).long().numpy()
            
            coco_instance_mask = torch.nn.functional.interpolate(
                torch.from_numpy(coco_instance_mask).unsqueeze(0).unsqueeze(0).double(), 
                scale_factor=factor, mode='nearest')[0, 0].numpy()
            
        return self.paste_anomalous_instance(cityscapes_image=np.array(cityscapes_image),
                                             cityscapes_target=np.array(cityscapes_target),
                                             coco_image=coco_image,
                                             coco_instance_mask=coco_instance_mask,
                                             ood_id=ood_id, id_id=id_id)
            
    def __getitem__(self, index):
        cityscapes_image, cityscapes_target = self.cityscapes_dataset[index]
        
        ood_index = np.random.randint(len(self.selected_coco_ids))
        coco_image, coco_target = self.coco_dataset[int(self.selected_coco_ids[ood_index])]
        
        mixed_image, mixed_target = self.create_mixed_image(cityscapes_image, cityscapes_target, coco_image, 
                                                            coco_target, ood_id=self.ood_id, id_id=self.id_id)
        
        if self.transform is not None:
            coco_image = self.transform(coco_image)
            mixed_image = self.transform(mixed_image)
        
        return coco_image, mixed_image, mixed_target        
        
    def __len__(self):
        return len(self.cityscapes_dataset)


class CityscapesWithCocoFeaturesDataset(Dataset):
    def __init__(self, cfg: CfgNode, device: torch.device):
        self.root = Path(cfg.SYSTEM.DINOV2_FEATURES_ROOT)
        self.dataset = CityscapesWithCocoDataset(cfg)
        
        if self.root.exists() and len(self.dataset) == len(list(self.patch_features_root.iterdir())):
            print(f"DINOv2 per-patch features have already been generated. Skipping...")
        else:
            print(f"Cityscapes per-patch features have not been generated. Generating...")
            print(f"Saving per-patch features to {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)
            
            dinov2 = torch.hub.load("facbookresearch/dinov2", cfg.BACKBONE.ARCHITECTURE).to(device=device)
            dinov2.eval()
            
            dataloader = DataLoader(self.dataset, batch_size=1, shuffle=False)
            with Progress() as progress:
                task = progress.add_task(f"Processing CityscapesWithCocoDataset", total=len(self.dataset))
                for i, (coco_image, mixed_image, mixed_target) in enumerate(dataloader):
                    progress.update(task, advance=1, 
                                    description=f"Generating per-patch features for CityscapesWithCocoDataset. Sample {i}/{len(self.dataset)}")
                    
                    with torch.no_grad():
                        coco_features = dinov2.forward_features(coco_image.to(device=device))["x_norm_patchtokens"]
                        mixed_image_features = dinov2.forward_features(mixed_image.to(device=device))["x_norm_patchtokens"]
                    
                    torch.save({"features" : coco_features.cpu().squeeze(), 
                                "labels" : mixed_target.cpu().squeeze()}, 
                               self.root / f"{Path(self.dataset.images[i]).stem}_coco.pth")
                    
                    torch.save({"features" : mixed_image_features.cpu().squeeze(),
                                "labels" : mixed_target.cpu().squeeze()}, 
                               self.root / f"{Path(self.dataset.images[i]).stem}.pth")

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        coco_features, labels = torch.load(f"{self.root / Path(self.dataset.images[index]).stem}_coco.pth", map_location="cpu").values()
        mixed_features, mixed_labels = torch.load(f"{self.root / Path(self.dataset.images[index]).stem}.pth", map_location="cpu").values()
        
        return coco_features, mixed_features, mixed_labels
        