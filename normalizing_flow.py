import torch
from torch import nn
import math
from abc import ABC, abstractmethod

class Diffeomorphism(nn.Module, ABC):
    def __init__(self):
        super(Diffeomorphism, self).__init__()
        
    @abstractmethod
    def forward(self, x):
        raise NotImplementedError
    
    @abstractmethod
    def inverse(self, y):
        raise NotImplementedError


class ActNorm(Diffeomorphism):
    def __init__(self, num_features):
        super(ActNorm, self).__init__()
        self.register_buffer("initialized", torch.tensor(False))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
    def initialize_parameters(self, x):
        with torch.no_grad():
            mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            std = torch.std(x, dim=(0, 2, 3), keepdim=True)
            
            self.log_scale.data.copy_(torch.log(std + 1e-6))
            self.bias.data.copy_(mean)            
        self.initialized.fill_(True)
    
    def forward(self, x):
        if not self.initialized and self.training:
            self.initialize_parameters(x)
        y = (x - self.bias) * torch.exp(-self.log_scale)
        return y, -torch.sum(self.log_scale.expand_as(x), dim=1)
    
    def inverse(self, y):
        x = y * torch.exp(self.log_scale) + self.bias
        return x
    

class RandomChannelWisePermutation(Diffeomorphism):
    def __init__(self, num_channels):
        super(RandomChannelWisePermutation, self).__init__()
        self.register_buffer("random_channel_permutation", torch.randperm(num_channels))
        self.register_buffer("random_channel_permutation_inv", torch.empty_like(self.random_channel_permutation))
        self.random_channel_permutation_inv[self.random_channel_permutation] = torch.arange(num_channels)
        
    def forward(self, x):
        B, C, H, W = x.shape
        return x[:, self.random_channel_permutation, :, :], torch.zeros(B, H, W, device=x.device, dtype=x.dtype)
    
    def inverse(self, y):
        return y[:, self.random_channel_permutation_inv, :, :]
    
    
class Subnet2d(nn.Module):
    def __init__(self, num_features, kernel_size):
        super(Subnet2d, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, 2*num_features, kernel_size, padding="same")
        )
        
        nn.init.zeros_(self.model[-1].weight)
        if self.model[-1].bias is not None:
            nn.init.zeros_(self.model[-1].bias)
            
    def forward(self, x):
        return torch.chunk(self.model(x), 2, dim=1)


class AffineCouplingLayer(Diffeomorphism):
    def __init__(self, num_features, conv2d_kernel_size=3):
        super(AffineCouplingLayer, self).__init__()
        self.num_features = num_features
        self.conv2d_kernel_size = conv2d_kernel_size
        
        self.subnet1 = Subnet2d(num_features // 2, conv2d_kernel_size)
        self.subnet2 = Subnet2d(num_features // 2, conv2d_kernel_size)
        
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        
        log_s1, b1 = self.subnet1(x1)
        log_s1 = 3.0 * torch.tanh(log_s1)
        y1 = torch.exp(log_s1) * x2 + b1
        
        log_s2, b2 = self.subnet2(y1)
        log_s2 = 3.0 * torch.tanh(log_s2)
        y2 = torch.exp(log_s2) * x1 + b2
        
        return torch.cat([y1, y2], dim=1), torch.sum(log_s1, dim=1) + torch.sum(log_s2, dim=1)
        
    def inverse(self, y):
        y1, y2 = y.chunk(2, dim=1)
        
        log_s2, b2 = self.subnet2(y1)
        x1 = (y2 - b2) * torch.exp(-log_s2)
        
        log_s1, b1 = self.subnet1(x1)
        x2 = (y1 - b1) * torch.exp(-log_s1)
        
        return torch.cat([x1, x2], dim=1)


class NormalizingFlow(Diffeomorphism):
    def __init__(self, num_features, num_steps=16, projection_head_dim=256):
        super(NormalizingFlow, self).__init__()
        self.num_features = num_features
        self.mu = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.log_sigma = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
        self.flow_steps: nn.ModuleList[Diffeomorphism] = nn.ModuleList()
        for i in range(num_steps):
            self.flow_steps.append(ActNorm(num_features))
            self.flow_steps.append(RandomChannelWisePermutation(num_features))
            self.flow_steps.append(AffineCouplingLayer(num_features, conv2d_kernel_size= 3 if i % 2 == 0 else 1))
                
        self.projection = nn.Conv2d(num_features, projection_head_dim, kernel_size=1, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape
        z = x
        log_abs_det_jacobian = torch.zeros(B, H, W, device=x.device, dtype=x.dtype)
        
        for flow_step in self.flow_steps:
            z, _log_abs_det_jacobian = flow_step(z)
            log_abs_det_jacobian += _log_abs_det_jacobian
        
        z_projected = (z - self.mu) * torch.exp(-self.log_sigma)
        z_projected = self.projection(z_projected)
           
        return z, z_projected, log_abs_det_jacobian
    
    def inverse(self, y):
        x = y
        for flow_step in reversed(self.flow_steps):
            x = flow_step.inverse(x)
        return x
    
    def log_density(self, z, log_abs_det_jacobian, return_anomaly_score=False):
        C = z.shape[1]
                         
        element_wise_log_pz = -0.5 * (((z - self.mu) * torch.exp(-self.log_sigma)) ** 2 + math.log(2 * math.pi) + 2 * self.log_sigma)
        
        log_pz = torch.sum(element_wise_log_pz, dim=1)  + log_abs_det_jacobian # (B, H, W)
        
        if not return_anomaly_score:
            return log_pz

        NPD_anomaly_score = -(1.0 / C) * log_pz
        return NPD_anomaly_score
    