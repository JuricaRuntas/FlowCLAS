import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import random
import argparse
from yacs.config import CfgNode
from tqdm import tqdm
from pathlib import Path

from normalizing_flow import NormalizingFlow
from logger import logger
from config import get_cfg_defaults
from datasets.features_dataset import (CityscapesWithCocoFeaturesDataset, RoadAnomalyFeaturesDataset, 
                                       FishyscapesLostAndFoundFeaturesDataset, FishyscapesStaticFeaturesDataset)
from utils import CosineLRWithLinearWarmup, calculate_eval_metrics, SupervisedContrastive2d, seed_worker, collate_fn_pad_coco_features

def eval(normalizing_flow: NormalizingFlow, datasets: list[Dataset], cfg: CfgNode, device: torch.device):
    normalizing_flow.eval()

    result = {}
    
    for dataset in datasets:
        labels = []
        anomaly_scores = []
        
        for img_features, target in tqdm(dataset, desc=f"Evaluating {dataset.name}"):
            H, W = img_features.shape[-2:]
            
            with torch.no_grad():
                z, z_projected, log_det_jacobian = normalizing_flow(img_features.to(device).unsqueeze(0))
                anomaly_score = normalizing_flow.log_density(z, log_det_jacobian, return_anomaly_score=True)
                
                anomaly_score = torch.nn.functional.interpolate(
                        anomaly_score.unsqueeze(1), size=target.shape[-2:], mode="bilinear", align_corners=False
                )
                
                target_vec = target.view(-1)
                
                anomaly_scores.append(anomaly_score.view(-1).cpu().numpy()[target_vec != cfg.DATASETS.IGNORE_LABEL])
                labels.append(target_vec[target_vec != cfg.DATASETS.IGNORE_LABEL].cpu().numpy())

        logger.info(f"Evaluating metrics for {dataset.name}...")
        labels = np.concatenate(labels, axis=0)
        anomaly_scores = np.concatenate(anomaly_scores, axis=0)
        
        AP, AUROC, FPR_95, TAU_FPR_95 = calculate_eval_metrics(anomaly_scores, labels)
        
        res = {}
        res["AP"] = 100 * AP
        res["AUROC"] = 100 * AUROC
        res["FPR@TPR95"] = 100 * FPR_95
        
        logger.info(f"Results for {dataset.name}: {res}")

        result[dataset.name] = res
        
        torch.cuda.empty_cache()

    return result


def train(normalizing_flow: NormalizingFlow, cfg: CfgNode, device: torch.device, resume_checkpoint_path: str = None):
    normalizing_flow.train()
    
    logger.info("Starting training...")
    logger.info(f"Total number of trainable parameters: {round(sum(p.numel() for p in normalizing_flow.parameters() if p.requires_grad) / 1e6, 2)}M")

    dataset = CityscapesWithCocoFeaturesDataset(cfg, device)
    dataloader = DataLoader(dataset, batch_size=cfg.NORMALIZING_FLOW.BATCH_SIZE, shuffle=True, 
                            num_workers=cfg.SYSTEM.NUM_WORKERS, worker_init_fn=seed_worker, 
                            collate_fn=collate_fn_pad_coco_features, pin_memory=True, drop_last=False)
    
    optimizer = torch.optim.AdamW(normalizing_flow.parameters(), lr=cfg.NORMALIZING_FLOW.LR, weight_decay=cfg.NORMALIZING_FLOW.WEIGHT_DECAY)    
    scheduler = CosineLRWithLinearWarmup(optimizer, cfg.NORMALIZING_FLOW.WARMUP_EPOCHS * len(dataloader), cfg.NORMALIZING_FLOW.NUM_EPOCHS * len(dataloader))
    contrastive_loss = SupervisedContrastive2d(temperature=cfg.NORMALIZING_FLOW.TEMPERATURE, ignore_label=cfg.DATASETS.IGNORE_LABEL)
        
    Path(cfg.SYSTEM.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    start_epoch = 0
    best_ap = 0.0
    epochs_since_improvement = 0
    patience = 30
    eval_every = 2
    
    if resume_checkpoint_path:
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint_path}")
        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)
        normalizing_flow.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        logger.info(f"Resumed training from epoch {start_epoch}")    
    
    for epoch in range(start_epoch, cfg.NORMALIZING_FLOW.NUM_EPOCHS):
        normalizing_flow.train()
        total_loss = 0.0

        logger.info(f"Epoch {epoch + 1}/{cfg.NORMALIZING_FLOW.NUM_EPOCHS}")

        for i, (padded_coco_features, mixed_image_features, mixed_target, coco_features_shape) in enumerate(tqdm(dataloader, unit="batch")):
            padded_coco_features = padded_coco_features.to(device)
            mixed_image_features = mixed_image_features.to(device)
            mixed_target = mixed_target.to(device)
            
            if cfg.NORMALIZING_FLOW.FEATURE_NOISE_STD > 0:
                noise = torch.randn_like(mixed_image_features) * cfg.NORMALIZING_FLOW.FEATURE_NOISE_STD
                mixed_image_features += noise
            
            z_mix, z_mix_projected, z_mix_log_abs_det_jacobian = normalizing_flow(mixed_image_features)
            NLL_loss = -normalizing_flow.log_density(z_mix, z_mix_log_abs_det_jacobian)[mixed_target == 0]

            z_out, z_out_projected, z_out_log_abs_det_jacobian = normalizing_flow(padded_coco_features)
            batch_size, _, H, W = z_out_projected.shape
            z_out_valid_mask = torch.zeros((batch_size, H, W), dtype=torch.bool, device=device)
            for b, (H_original, W_original) in enumerate(coco_features_shape):
                z_out_valid_mask[b, :H_original, :W_original] = 1
            
            contrastive = contrastive_loss(z_mix_projected, mixed_target, z_out_projected, z_out_valid_mask)
            nll = NLL_loss.mean()
            loss = cfg.NORMALIZING_FLOW.ALPHA * nll + contrastive
                        
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(normalizing_flow.parameters(), cfg.NORMALIZING_FLOW.MAX_NORM)
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
            
            if not torch.isfinite(loss):
                logger.error(f"Loss is NaN or Inf at batch {i+1} in epoch {epoch+1}. Stopping training...")
             
            if (i + 1) % 10 == 0:
                logger.info(f"Batch {i+1}/{len(dataloader)} - NLL: {nll.item():.4f} | Contrastive: {contrastive.item():.4f} | Total Loss: {loss.item():.4f}")

        torch.cuda.empty_cache()

        logger.info(f"Epoch {epoch+1} average loss: {total_loss / len(dataloader):.4f}")
        logger.info("Saving model checkpoint...")
        torch.save({"epoch": epoch, "model_state_dict": normalizing_flow.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict()},
                   f"{cfg.SYSTEM.OUTPUT_DIR}/checkpoint_latest.pth")
        
        if (epoch + 1) % eval_every == 0:
            eval_dataset = RoadAnomalyFeaturesDataset(cfg, device)
            ap = eval(normalizing_flow, [eval_dataset], cfg, device)[eval_dataset.name]["AP"]
            logger.info(f"AP on Road Anomaly at epoch {epoch+1}: {ap:.2f}")
            
            if ap > best_ap:
                best_ap = ap
                epochs_since_improvement = 0
                logger.info(f"New best AP on Road Anomaly: {best_ap:.2f}.")
                torch.save({"epoch": epoch, "model_state_dict": normalizing_flow.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()},
                            f"{cfg.SYSTEM.OUTPUT_DIR}/checkpoint_best_{best_ap}.pth")
            else:
                epochs_since_improvement += eval_every
                logger.info(f"No improvement in AP for {epochs_since_improvement} epochs.")
            
            if epochs_since_improvement >= patience:
                logger.warning(f"Early stopping triggered at epoch {epoch+1}")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint.")
    parser.add_argument("--weights", type=str, default=None, help="Path to the model weights for evaluation or resume training.")
    args = parser.parse_args()
    
    if not args.train and not args.eval:
        raise ValueError("You must specify either --train or --eval.")
    
    cfg = get_cfg_defaults()
    cfg.freeze()
    
    logger.info(cfg)

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

        normalizing_flow.load_state_dict(torch.load(args.weights, map_location=device, weights_only=False)["model_state_dict"])
        road_driving_ood_datasets = [FishyscapesStaticFeaturesDataset(cfg, device), 
                                     RoadAnomalyFeaturesDataset(cfg, device), 
                                     FishyscapesLostAndFoundFeaturesDataset(cfg, device)]
        eval(normalizing_flow, road_driving_ood_datasets, cfg, device)
    elif args.train:
        resume_checkpoint = args.weights if args.resume else None
        if args.resume and args.weights is None:
            raise ValueError("Path to weights must be specified when using --resume.")
        train(normalizing_flow, cfg, device, resume_checkpoint)
    