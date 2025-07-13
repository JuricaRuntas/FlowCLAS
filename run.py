import numpy as np
import torch
from torch.utils.data import DataLoader
import random
import argparse
from yacs.config import CfgNode
from tqdm import tqdm
from pathlib import Path

from normalizing_flow import NormalizingFlow

from config import get_cfg_defaults
from datasets.cityscapes_with_coco import CityscapesWithCocoFeaturesDataset
from datasets.ood_datasets import RoadAnomaly, FishyscapesLostAndFound, FishyscapesStatic
from utils import CosineLRWithLinearWarmup, calculate_eval_metrics, InfoNCE2d, seed_worker, collate_fn_pad_coco_features

def eval(normalizing_flow: NormalizingFlow, cfg: CfgNode, device: torch.device):
    normalizing_flow.eval()
    
    dinov2 = torch.hub.load("facebookresearch/dinov2", cfg.BACKBONE.ARCHITECTURE).to(device=device)
    dinov2.eval()
    
    road_driving_ood_datasets = [FishyscapesStatic(cfg), RoadAnomaly(cfg), FishyscapesLostAndFound(cfg)]

    for dataset in road_driving_ood_datasets:
        labels = []
        anomaly_scores = []
        
        for img, target, img_path, target_path in tqdm(dataset, desc=f"Evaluating {dataset.name}"):
            H, W = img.shape[-2:]
            
            if not (H % cfg.BACKBONE.PATCH_SIZE == 0 and W % cfg.BACKBONE.PATCH_SIZE == 0):
                raise ValueError(f"Image dimensions {H}x{W} must be divisible by patch size {cfg.BACKBONE.PATCH_SIZE}.")
                                 
            H //= cfg.BACKBONE.PATCH_SIZE
            W //= cfg.BACKBONE.PATCH_SIZE
            
            with torch.no_grad():
                img_features = dinov2.forward_features(img.unsqueeze(0).to(device))["x_norm_patchtokens"]
                img_features = img_features.view(-1, H, W, cfg.BACKBONE.EMBED_DIM).permute(0, 3, 1, 2)
                
                z, z_projected, log_det_jacobian = normalizing_flow(img_features)
                anomaly_score = normalizing_flow.log_density(z, log_det_jacobian, return_anomaly_score=True)
                
                anomaly_score = torch.nn.functional.interpolate(
                        anomaly_score.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False
                )
                
                target_vec = target.view(-1)
                
                anomaly_scores.append(anomaly_score.view(-1).cpu().numpy()[target_vec != cfg.DATASETS.IGNORE_LABEL])
                labels.append(target_vec[target_vec != cfg.DATASETS.IGNORE_LABEL].cpu().numpy())        
            
        print("Evaluating metrics...")
        labels = np.concatenate(labels, axis=0)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        
        AP, AUROC, FPR_95, TAU_FPR_95 = calculate_eval_metrics(anomaly_scores, labels)
        
        res = {}
        res["AP"] = 100 * AP
        res["AUROC"] = 100 * AUROC
        res["FPR@TPR95"] = 100 * FPR_95
        
        print(f"Results for {dataset.name}: {res}")


def train(normalizing_flow: NormalizingFlow, cfg: CfgNode, device: torch.device):
    normalizing_flow.train()
    
    print("Starting training...")
    print(f"Total number of trainable parameters: {round(sum(p.numel() for p in normalizing_flow.parameters() if p.requires_grad) / 1e6, 2)}M")
    
    optimizer = torch.optim.AdamW(normalizing_flow.parameters(), lr=cfg.NORMALIZING_FLOW.LR, weight_decay=cfg.NORMALIZING_FLOW.WEIGHT_DECAY)
    scheduler = CosineLRWithLinearWarmup(optimizer, cfg.NORMALIZING_FLOW.WARMUP_EPOCHS, cfg.NORMALIZING_FLOW.NUM_EPOCHS)
    contrastive_loss = InfoNCE2d(max_anchors_per_class=cfg.NORMALIZING_FLOW.MAX_ANCHORS_PER_CLASS,
                                 temperature=cfg.NORMALIZING_FLOW.TEMPERATURE, 
                                 ignore_label=cfg.DATASETS.IGNORE_LABEL)
    
    dataset = CityscapesWithCocoFeaturesDataset(cfg, device)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=cfg.SYSTEM.NUM_WORKERS, 
                            worker_init_fn=seed_worker, collate_fn=collate_fn_pad_coco_features,
                            pin_memory=True, drop_last=False)
    
    for epoch in range(cfg.NORMALIZING_FLOW.NUM_EPOCHS):
        normalizing_flow.train()
        total_loss = 0.0
        
        print(f"Epoch {epoch + 1}/{cfg.NORMALIZING_FLOW.NUM_EPOCHS}")
        
        for i, (padded_coco_features, mixed_image_features, mixed_target, coco_features_shape) in enumerate(tqdm(dataloader, unit="batch")):
            padded_coco_features = padded_coco_features.to(device)
            mixed_image_features = mixed_image_features.to(device)
            mixed_target = mixed_target.to(device)
            
            z_mix, z_mix_projected, z_mix_log_abs_det_jacobian = normalizing_flow(mixed_image_features)
            NLL_loss = -normalizing_flow.log_density(z_mix, z_mix_log_abs_det_jacobian)[mixed_target == 0]
            
            z_out, z_out_projected, z_out_log_abs_det_jacobian = normalizing_flow(padded_coco_features)
            
            batch_size, _, H, W = z_out_projected.shape
            z_out_valid_mask = torch.zeros((batch_size, H, W), dtype=torch.bool, device=device)
            for b, (H_original, W_original) in enumerate(coco_features_shape):
                z_out_valid_mask[b, :H_original, :W_original] = 1
                                    
            loss = cfg.NORMALIZING_FLOW.ALPHA * NLL_loss.mean() + contrastive_loss(z_mix_projected, mixed_target, z_out_projected, z_out_valid_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
            
            if (i + 1) % 10 == 0:
                print(f"Batch {i+1}/{len(dataloader)} - Loss: {loss.item():.4f}")
            
        print(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader):.4f}")
        print("Saving model checkpoint...")
        Path(cfg.SYSTEM.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)   
        torch.save(normalizing_flow.state_dict(), f"{cfg.SYSTEM.OUTPUT_DIR}/normalizing_flow_epoch_{epoch+1}.pth")
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--weights", type=str, default=None, help="Path to the model weights for evaluation.")
    args = parser.parse_args()
    
    if not args.train and not args.eval:
        raise ValueError("You must specify either --train or --eval.")
    
    cfg = get_cfg_defaults()
    cfg.freeze()
    
    np.random.seed(cfg.SYSTEM.SEED)
    torch.manual_seed(cfg.SYSTEM.SEED)
    random.seed(cfg.SYSTEM.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    normalizing_flow = NormalizingFlow(cfg.NORMALIZING_FLOW.NUM_FEATURES,
                                       cfg.NORMALIZING_FLOW.NUM_STEPS,
                                       cfg.NORMALIZING_FLOW.PROJECTION_HEAD_DIM).to(device)
        
    if args.eval:
        if args.weights is None:
            raise ValueError("Path to weights must be specified for evaluation.")
        
        normalizing_flow.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
        eval(normalizing_flow, cfg, device)
    elif args.train:
        train(normalizing_flow, cfg, device)
    