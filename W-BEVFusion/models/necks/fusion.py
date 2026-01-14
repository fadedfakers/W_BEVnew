import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import BEVConfig as cfg

class CrossModalAttention(nn.Module):
    """
    Fuses LiDAR BEV features with Image features using Cross-Attention.
    Fixed: Dynamic input shapes and robust residual connections.
    """
    def __init__(self, lidar_channels: int = 256, img_channels: int = 2048, embed_dims: int = 256):
        super().__init__()
        self.embed_dims = embed_dims
        
        # Projections
        self.lidar_proj = nn.Conv2d(lidar_channels, embed_dims, kernel_size=1)
        self.img_proj = nn.Conv2d(img_channels, embed_dims, kernel_size=1)
        
        # LayerNorms
        self.ln_q = nn.LayerNorm(embed_dims)
        self.ln_kv = nn.LayerNorm(embed_dims)
        
        # Multi-Head Attention
        self.attention = nn.MultiheadAttention(embed_dims, num_heads=8, batch_first=True, dropout=0.1)
        
        # Output layers
        self.output_conv = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True)
        )
        
        # [Fix] Use dynamic config sizes instead of hardcoded 128x256
        self.bev_pos_embed = nn.Parameter(
            torch.randn(1, embed_dims, cfg.GRID_H, cfg.GRID_W) * 0.01
        )

    def forward(self, lidar_bev: torch.Tensor, img_features: torch.Tensor) -> torch.Tensor:
        """
        lidar_bev: (B, C_lidar, H_bev, W_bev)
        img_features: (B, C_img, H_img, W_img)
        """
        B, C, H, W = lidar_bev.shape
        
        # 1. Project inputs
        # (B, embed_dims, H, W)
        lidar_embed = self.lidar_proj(lidar_bev) 
        
        # Add Positional Embedding (Broadcasting works if batch size > 1)
        q_feat = lidar_embed + self.bev_pos_embed 
        
        # Project Image Features
        kv_feat = self.img_proj(img_features)
        
        # 2. Flatten for Attention
        # Query: LiDAR BEV pixels -> (B, H*W, embed_dims)
        q = q_feat.flatten(2).permute(0, 2, 1) 
        
        # Key/Value: Image pixels -> (B, H_img*W_img, embed_dims)
        kv = kv_feat.flatten(2).permute(0, 2, 1) 
        
        # LayerNorm before Attention (Pre-Norm)
        q = self.ln_q(q)
        kv = self.ln_kv(kv)
        
        # 3. Cross-Attention
        # Update LiDAR features based on Image context
        attn_out, _ = self.attention(query=q, key=kv, value=kv)
        
        # 4. Reshape back to BEV grid
        fused = attn_out.permute(0, 2, 1).view(B, self.embed_dims, H, W)
        
        # 5. Residual Connection & Output
        # [Fix] Ensure we add the projected lidar_embed (which matches embed_dims)
        # instead of raw lidar_bev (which might differ in channels)
        fused_bev = self.output_conv(fused + lidar_embed)
        
        return fused_bev