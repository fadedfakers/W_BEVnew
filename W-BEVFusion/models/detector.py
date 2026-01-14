import torch
import torch.nn as nn
from typing import Dict, Any, List

from models.backbones.wavelet_img import WaveletResNet
from models.backbones.pillar_lidar import PillarEncoder
from models.necks.fusion import CrossModalAttention
from models.heads.multitask_head import BEVMultiHead
from configs.config import BEVConfig as cfg

class WBEVFusionNet(nn.Module):
    """
    Wavelet-Enhanced BEV Fusion Network for Railway Obstacle Detection.
    
    Pipeline:
    Image (RGB) -> WaveletResNet (Backbone) ---|
                                              |--> CrossModalAttention (Neck) -> BEVMultiHead (Heads)
    Lidar (PCD) -> PillarEncoder (Backbone)---|
    """
    def __init__(self, config: Any = cfg):
        super().__init__()
        
        # 1. Image Backbone (Wavelet-Enhanced)
        self.img_backbone = WaveletResNet(pretrained=True)
        
        # 2. LiDAR Backbone (Pillar-based)
        # Note: input channels = 4 (x, y, z, i), output = 256
        self.lidar_backbone = PillarEncoder(in_channels=4, out_channels=256)
        
        # 3. Fusion Neck (Cross-Modal Attention)
        # Query channels = 256 (Lidar), Key/Value = 2048 (ResNet Layer 4)
        self.fusion_neck = CrossModalAttention(
            lidar_channels=256, 
            img_channels=2048, 
            embed_dims=256
        )
        
        # 4. Multi-Task Heads
        self.head = BEVMultiHead(in_channels=256, num_classes=config.NUM_CLASSES)

    def forward(self, images: torch.Tensor, points_list: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for W-BEVFusion.
        
        Args:
            images: (B, 3, H_img, W_img)
            points_list: List of (N, 4) Lidar points
        Returns:
            Dict: {'cls_pred', 'box_pred', 'mask_pred'}
        """
        # 1. Extract Image Features
        # WaveletResNet returns [f2, f3, f4]
        img_feats = self.img_backbone(images)
        # Use the strongest features from Layer 4 (2048 channels)
        f4 = img_feats[-1] 
        
        # 2. Extract LiDAR BEV Features
        # Output: (B, 256, 128, 256)
        lidar_bev = self.lidar_backbone(points_list)
        
        # 3. Cross-Modal Fusion
        # LiDAR BEV acts as Query to attend Image Features
        fused_bev = self.fusion_neck(lidar_bev, f4)
        
        # 4. Multi-Task Prediction
        predictions = self.head(fused_bev)
        
        return predictions
