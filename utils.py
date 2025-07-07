import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from sklearn.metrics import roc_curve, auc, average_precision_score

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
    
    
def calculate_eval_metrics(conf: np.ndarray, labels: np.ndarray) -> tuple[float, float, float, float]:
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