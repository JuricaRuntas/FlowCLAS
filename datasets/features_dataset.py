import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from yacs.config import CfgNode
from rich.progress import Progress
from datasets.cityscapes_with_coco import CityscapesWithCocoDataset
from datasets.ood_datasets import RoadAnomaly, FishyscapesLostAndFound, FishyscapesStatic

def generate_features(dataset: Dataset, cfg: CfgNode, device: torch.device, cityscapes: bool = False):
    """
    Generates DINOv2 per-patch features for the given dataset and
    saves them to disk.
    
    Args:
        dataset (Dataset): The dataset for which to generate features.
        cfg (CfgNode): Configuration node containing settings.
        device (torch.device): Device to run the feature extraction on.
        cityscapes (bool): If True, generates features for Cityscapes dataset.
    """
    root = Path(cfg.SYSTEM.DINOV2_FEATURES_ROOT) / dataset.name
    patch_size = cfg.BACKBONE.PATCH_SIZE
    embed_dim = cfg.BACKBONE.EMBED_DIM
        
    if root.exists():
        print(f"DINOv2 per-patch features for dataset {dataset.name} have already been generated. Skipping...")
    else:
        print(f"{dataset.name} per-patch features have not been generated. Generating...")
        print(f"Saving per-patch features to {root}")
        root.mkdir(parents=True, exist_ok=True)

        dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.BACKBONE.ARCHITECTURE).to(device=device)
        dinov2.eval()

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        with Progress() as progress:
            task = progress.add_task(f"Processing {dataset.name}", total=len(dataset))
            
            if cityscapes:
                for i, (aux_image, image, target) in enumerate(dataloader):
                    progress.update(task, advance=1, description=f"Generating per-patch features for {dataset.name}. Sample {i}/{len(dataset)}")
                    
                    _, _, H_aux, W_aux = aux_image.shape
                    _, _, H_image, W_image = image.shape

                    if not (H_aux % patch_size == 0 and W_aux % patch_size == 0 and
                            H_image % patch_size == 0 and W_image % patch_size == 0):
                        raise ValueError(f"Image dimensions {H_aux}x{W_aux} and {H_image}x{W_image} must be divisible by patch size {patch_size}.")
                                        
                    H_aux //= patch_size
                    W_aux //= patch_size
                    H_image //= patch_size
                    W_image //= patch_size

                    with torch.no_grad():
                        aux_features = dinov2.forward_features(aux_image.to(device=device))["x_norm_patchtokens"]
                        image_features = dinov2.forward_features(image.to(device=device))["x_norm_patchtokens"]

                    torch.save({"features" : aux_features.view(-1, H_aux, W_aux, embed_dim).permute(0, 3, 1, 2).cpu().squeeze(), 
                                "target" : target.cpu().squeeze()}, root / f"{Path(dataset.cityscapes_dataset.images[i]).stem}_coco.pth")

                    torch.save({"features" : image_features.view(-1, H_image, W_image, embed_dim).permute(0, 3, 1, 2).cpu().squeeze(), 
                                "target" : target.cpu().squeeze()}, root / f"{Path(dataset.cityscapes_dataset.images[i]).stem}.pth")
            else:
                for i, (image, target, _, _) in enumerate(dataloader):
                    progress.update(task, advance=1, description=f"Generating per-patch features for {dataset.name}. Sample {i}/{len(dataset)}")
                    
                    _, _, H, W = image.shape

                    if not (H % patch_size == 0 and W % patch_size == 0):
                        raise ValueError(f"Image dimensions {H}x{W} must be divisible by patch size {patch_size}.")
                    
                    H //= patch_size
                    W //= patch_size

                    with torch.no_grad():
                        features = dinov2.forward_features(image.to(device=device))["x_norm_patchtokens"]

                    torch.save({"features" : features.view(-1, H, W, embed_dim).permute(0, 3, 1, 2).cpu().squeeze(), 
                                "target" : target.cpu().squeeze()}, root / f"{Path(dataset.images[i]).stem}.pth")
                
                
class CityscapesWithCocoFeaturesDataset(Dataset):
    def __init__(self, cfg: CfgNode, device: torch.device):
        self.dataset = CityscapesWithCocoDataset(cfg)
        self.root = Path(cfg.SYSTEM.DINOV2_FEATURES_ROOT) / self.name
        
        generate_features(self.dataset, cfg, device, cityscapes=True)        
    
    @property
    def name(self):
        return self.dataset.name

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        coco_features, target = torch.load(f"{self.root / Path(self.dataset.cityscapes_dataset.images[index]).stem}_coco.pth", 
                                           map_location="cpu", weights_only=False).values()
        mixed_features, mixed_target = torch.load(f"{self.root / Path(self.dataset.cityscapes_dataset.images[index]).stem}.pth", 
                                                  map_location="cpu", weights_only=False).values()
        return coco_features, mixed_features, mixed_target
        
class RoadAnomalyFeaturesDataset(Dataset):
    def __init__(self, cfg: CfgNode, device: torch.device):
        self.dataset = RoadAnomaly(cfg)
        self.root = Path(cfg.SYSTEM.DINOV2_FEATURES_ROOT) / self.name
        
        generate_features(self.dataset, cfg, device)
        
    @property
    def name(self):
        return self.dataset.name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, target = torch.load(self.root / f"{Path(self.dataset.images[index]).stem}.pth", map_location="cpu", weights_only=False).values()
        return features, target
     
class FishyscapesLostAndFoundFeaturesDataset(Dataset):
    def __init__(self, cfg: CfgNode, device: torch.device):
        self.dataset = FishyscapesLostAndFound(cfg)
        self.root = Path(cfg.SYSTEM.DINOV2_FEATURES_ROOT) / self.name
        
        generate_features(self.dataset, cfg, device)
        
    @property
    def name(self):
        return self.dataset.name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, target = torch.load(self.root / f"{Path(self.dataset.images[index]).stem}.pth", map_location="cpu", weights_only=False).values()
        return features, target 
 
class FishyscapesStaticFeaturesDataset(Dataset):
    def __init__(self, cfg: CfgNode, device: torch.device):
        self.dataset = FishyscapesStatic(cfg)
        self.root = Path(cfg.SYSTEM.DINOV2_FEATURES_ROOT) / self.name

        generate_features(self.dataset, cfg, device)
    
    @property
    def name(self):
        return self.dataset.name
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        features, target = torch.load(self.root / f"{Path(self.dataset.images[index]).stem}.pth", map_location="cpu", weights_only=False).values()
        return features, target
    