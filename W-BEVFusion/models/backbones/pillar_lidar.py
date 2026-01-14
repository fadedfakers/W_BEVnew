import torch
import torch.nn as nn
from typing import List, Tuple
from configs.config import BEVConfig as cfg

class PillarEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels
        self.pfn = nn.Sequential(
            nn.Linear(in_channels + 6, 64),
            nn.BatchNorm1d(64, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channels),
            nn.BatchNorm1d(out_channels, eps=1e-3),
            nn.ReLU(inplace=True)
        )

    def forward(self, points_list):
        B = len(points_list)
        if B == 0:
            return torch.zeros((B, self.out_channels, cfg.GRID_H, cfg.GRID_W)).to(points_list[0].device)

        device = points_list[0].device
        # 初始化 BEV Map (Float32)
        bev_map = torch.zeros((B, self.out_channels, cfg.GRID_H, cfg.GRID_W), device=device)
        
        for b_idx, points in enumerate(points_list):
            if points.shape[0] == 0:
                continue
                
            # 1. Coordinate to Grid Indices
            gx = ((points[:, 0] - cfg.X_RANGE[0]) / cfg.VOXEL_SIZE).long()
            gy = ((points[:, 1] - cfg.Y_RANGE[0]) / cfg.VOXEL_SIZE).long()
            
            # Filter valid indices
            mask = (gx >= 0) & (gx < cfg.GRID_W) & (gy >= 0) & (gy < cfg.GRID_H)
            points = points[mask]
            gx, gy = gx[mask], gy[mask]
            
            if points.shape[0] == 0:
                continue
                
            # 2. Augmented Features calculation
            # Local offsets to pillar center
            cx = gx.float() * cfg.VOXEL_SIZE + cfg.X_RANGE[0] + cfg.VOXEL_SIZE/2
            cy = gy.float() * cfg.VOXEL_SIZE + cfg.Y_RANGE[0] + cfg.VOXEL_SIZE/2
            dx = points[:, 0] - cx
            dy = points[:, 1] - cy
            
            # Augmented features: [x, y, z, i, dx, dy, dz=0, cx, cy, cz=0]
            aug_pts = torch.cat([points, dx.unsqueeze(1), dy.unsqueeze(1), 
                                torch.zeros_like(dx).unsqueeze(1),
                                cx.unsqueeze(1), cy.unsqueeze(1),
                                torch.zeros_like(dx).unsqueeze(1)], dim=1)
            
            # Forward Pass (AMP 模式下，这里输出可能是 Float16)
            feat = self.pfn(aug_pts)
            feat = torch.clamp(feat, min=-100, max=100) 
            
            # === Mean Pooling Logic ===
            indices = gy * cfg.GRID_W + gx
            
            total_pillars = cfg.GRID_H * cfg.GRID_W
            
            # 容器默认是 Float32，这对于累加操作是必要的，防止溢出
            pillar_feat = torch.zeros((total_pillars, self.out_channels), device=device)
            pillar_counts = torch.zeros((total_pillars, 1), device=device)
            
            # 【关键修复】将 feat 强转为 pillar_feat 的类型 (Float32)
            pillar_feat.index_add_(0, indices, feat.to(pillar_feat.dtype))
            
            # 计数
            ones = torch.ones((feat.shape[0], 1), device=device)
            pillar_counts.index_add_(0, indices, ones)
            
            # 求平均
            pillar_feat = pillar_feat / (pillar_counts + 1e-6)
            
            # 填回 Map
            bev_map[b_idx] = pillar_feat.view(cfg.GRID_H, cfg.GRID_W, self.out_channels).permute(2, 0, 1)
            
        return bev_map