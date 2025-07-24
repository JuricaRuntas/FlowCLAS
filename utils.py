import torch
import numpy as np
import random
import torch.nn.functional as F
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import roc_curve, auc, average_precision_score
from typing import Tuple
from datasets.transforms import pad_to_shape

class CosineLRWithLinearWarmup(LRScheduler):
    def __init__(self, optimizer: Optimizer, num_warmup_steps: int, 
                 num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1):
        """
        Creates a learning rate scheduler with a linear warmup phase followed by a cosine decay.

        Args:
            optimizer (Optimizer): PyTorch optimizer
            num_warmup_steps (int): Number of warmup steps
            num_training_steps (int): Total number of training steps
            num_cycles (float): Number of cosine cycles in the decay phase (default 0.5 = one half-cycle)
            last_epoch (int): Last epoch index for resuming (default -1)
        """
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        current_step = self.last_epoch
        if current_step < self.num_warmup_steps:
            return [base_lr * (current_step / max(1, self.num_warmup_steps)) for base_lr in self.base_lrs]
        
        progress = (current_step - self.num_warmup_steps) / max(1, self.num_training_steps - self.num_warmup_steps)
        cosine_decay = max(0.0, 0.5 * (1 + np.cos(2 * np.pi * self.num_cycles * progress)))
        return [base_lr * cosine_decay for base_lr in self.base_lrs]
    
    
def calculate_eval_metrics(conf: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Calculates pixel-level evaluation metrics.
    
    Args:
        conf (np.ndarray): Confidence scores for each pixel.
        labels (np.ndarray): Ground truth labels for each pixel.
    
    Returns:
        tuple: A tuple containing AP, AUROC, FPR@TPR95, threshold at TPR=95%.
    """
    
    fpr, tpr, threshold = roc_curve(labels, conf)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    threshold = np.array(threshold)

    AP = average_precision_score(labels, conf)
    AUROC = auc(fpr, tpr)
    FPR_95 = fpr[tpr >= 0.95][0]
    TAU_FPR_95 = threshold[tpr >= 0.95][0]
    
    return AP, AUROC, FPR_95, TAU_FPR_95


class SupervisedContrastive2d(nn.Module):
    """
    SupervisedContrastive2d loss for 2D (spatial) features.
    """
    def __init__(self, temperature: float = 1.0, ignore_label: int = 255):
        """
        Args:
            temperature (float): Temperature parameter for scaling.
            ignore_label (int): Label to ignore in the loss computation.
        """
        super(SupervisedContrastive2d, self).__init__()
        self.temperature = temperature
        self.ignore_label = ignore_label
        
    def construct_sets(self, z_mix, y_mix, z_out, z_out_valid_mask=None):
        batch_size, embedding_dim, H_mix, W_mix = z_mix.shape
        _, _, H_out, W_out = z_out.shape
        device = z_mix.device 

        z_mix = z_mix.view(batch_size, H_mix * W_mix, embedding_dim)
        y_mix = y_mix.view(batch_size, H_mix * W_mix)
        
        in_embeddings = z_mix[(y_mix == 0) & (y_mix != self.ignore_label)]
        out_embeddings = z_mix[y_mix == 1]
        
        if z_out_valid_mask is not None:
            z_out = z_out.permute(0, 2, 3, 1) # (B, H_out, W_out, C)
            z_out = z_out[z_out_valid_mask] # (B*H_out*W_out, C)
        else:
            z_out = z_out.view(batch_size*H_out*W_out, embedding_dim)
        
        N = min(in_embeddings.shape[0], out_embeddings.shape[0], z_out.shape[0])
        
        A_in = in_embeddings[torch.randperm(in_embeddings.shape[0])][:N]
        A_ood = out_embeddings[torch.randperm(out_embeddings.shape[0])][:N]
        
        B_ood = z_out[torch.randperm(z_out.shape[0])][:N]
        
        # anchor set
        A = torch.cat([A_in, A_ood], dim=0)
        A_labels = torch.cat([torch.zeros(A_in.shape[0], device=device),
                              torch.ones(A_ood.shape[0], device=device)], dim=0)
        
        # contrast set
        C = torch.cat([A_in, A_ood, B_ood], dim=0)
        C_labels = torch.cat([torch.zeros(A_in.shape[0], device=device),
                              torch.ones(A_ood.shape[0], device=device),
                              torch.ones(B_ood.shape[0], device=device)], dim=0)
        
        return A, A_labels, C, C_labels
        
    def forward(self, z_mix, y_mix, z_out, z_out_valid_mask=None):
        """
        Args:
            z_mix (torch.Tensor): Mixed features.
            y_mix (torch.Tensor): Labels for mixed features.
            z_out (torch.Tensor): Outlier features.
            z_out_valid_mask (torch.Tensor, optional): Mask for valid outlier features since they may have been padded.
        Returns:
            torch.Tensor: Computed SupervisedContrastive2d loss.
        """
        
        z_mix, z_out = F.normalize(z_mix, p=2, dim=1, eps=1e-6), F.normalize(z_out, p=2, dim=1, eps=1e-6)
        A, A_labels, C, C_labels = self.construct_sets(z_mix, y_mix, z_out, z_out_valid_mask)
        
        # (num_anchors, embed_dim) x (num_contrast, embed_dim) -> (num_anchors, num_contrast) 
        # cosine similarity matrix
        logits = torch.mm(A, C.T) / self.temperature
        
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()
        
        # A_labels[i] == C_labels[j] <=> (class(i) == class(j)) <=> ((i, j) is a positive pair)
        # mask the diagonal to avoid self-comparison
        diagonal_mask = torch.ones(A_labels.shape[0], C_labels.shape[0], dtype=torch.bool, device=A.device)
        # only mask the first 2N positions where A and C overlap (A_in and A_ood appear in both)
        diagonal_mask[:A_labels.shape[0], :A_labels.shape[0]] = ~torch.eye(A_labels.shape[0], dtype=torch.bool, device=A.device)
        positive_mask = torch.eq(A_labels[:, None], C_labels[None, :]) & diagonal_mask
        
        # average loss per anchor
        return - ((F.log_softmax(logits, dim=1) * positive_mask).sum(dim=1) / positive_mask.sum(dim=1).clamp(min=1)).mean()

    
def seed_worker(worker_id):
    """
    Seed worker function for DataLoader to ensure reproducibility.
    
    Args:
        worker_id (int): Worker ID.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
def collate_fn_pad_coco_features(batch):
    """
    Since COCO images and therefore their features can have different sizes,
    this function pads the features to the maximum height and width in the batch.
    """
    coco_features, mixed_image_features, mixed_target = zip(*batch)
    
    max_H = max([f.shape[-2] for f in coco_features])
    max_W = max([f.shape[-1] for f in coco_features])
    
    padded_coco_features = torch.stack([pad_to_shape(f, (max_H, max_W)) for f in coco_features], dim=0)
    mixed_features = torch.stack(mixed_image_features, dim=0)
    mixed_target = torch.stack(mixed_target, dim=0)
    coco_features_shape = [(f.shape[-2], f.shape[-1]) for f in coco_features]
    
    return padded_coco_features, mixed_features, mixed_target, coco_features_shape
