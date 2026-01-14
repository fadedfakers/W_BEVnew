import torch
import torch.nn as nn
import math
from typing import Dict
from configs.config import BEVConfig as cfg

class BEVMultiHead(nn.Module):
    """
    Multi-Task Prediction Head for W-BEVFusion.
    Fixed: Removed Sigmoid for numerical stability (use BCEWithLogitsLoss).
    Fixed: Added weight initialization for stable convergence.
    """
    def __init__(self, in_channels: int = 256, num_classes: int = cfg.NUM_CLASSES):
        super().__init__()
        
        # --- Branch 1: Detection Subnet (RetinaNet-style) ---
        # Shared convolution layers
        self.det_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        
        # Heatmap Head
        self.cls_head = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # Box Regression Head
        self.reg_head = nn.Conv2d(in_channels, cfg.BOX_CODE_SIZE, kernel_size=1)

        # --- Branch 2: Segmentation Head (FCN-style) ---
        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1) 
            # [Fix] Removed Sigmoid. Output raw logits for BCEWithLogitsLoss.
        )
        
        self.init_weights()

    def init_weights(self):
        # Specific initialization for Focal Loss stability
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize classification head bias to -2.19 (prob ~ 0.1)
        # This prevents initial loss from exploding due to heavy background imbalance
        prior_prob = 0.1
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_head.bias, bias_value)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Fused BEV feature map (B, 256, H, W)
        """
        det_feats = self.det_conv(x)
        
        cls_pred = self.cls_head(det_feats)
        box_pred = self.reg_head(det_feats)
        
        mask_pred = self.seg_head(x)
        
        return {
            'cls_pred': cls_pred,   # Logits
            'box_pred': box_pred,   # Raw Regress Values
            'mask_pred': mask_pred  # Logits
        }