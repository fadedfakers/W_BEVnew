import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import Dict, Tuple

def decode_boxes(predictions: Dict[str, torch.Tensor], K: int = 50, threshold: float = 0.3) -> torch.Tensor:
    cls_pred = predictions['cls_pred']
    box_pred = predictions['box_pred']
    B, C, H, W = cls_pred.shape
    
    # 1. 峰值提取 (Local Maxima)
    scores = torch.sigmoid(cls_pred)
    # 通过 max_pool2d 找到局部最大值点
    keep = (F.max_pool2d(scores, kernel_size=3, stride=1, padding=1) == scores)
    scores = scores * keep.float()
    
    # 2. 提取 Top-K
    scores_flat = scores.view(B, -1)
    topk_scores, topk_indices = torch.topk(scores_flat, K)
    
    # 计算网格坐标和类别
    topk_classes = (topk_indices // (H * W)).int()
    topk_indices_spatial = topk_indices % (H * W) # 空间位置索引
    topk_ys = (topk_indices_spatial // W).int()
    topk_xs = (topk_indices_spatial % W).int()
    
    # 3. 获取回归值 [dx, dy, dz, w, l, h, sin, cos]
    box_pred_flat = box_pred.view(B, 8, -1)
    # 在空间索引上进行 gather
    reg_selected = torch.gather(box_pred_flat, 2, topk_indices_spatial.unsqueeze(1).long().expand(-1, 8, -1))
    
    # 使用 Sigmoid 归一化偏移量到 0~1，防止预测出格
    dx = torch.sigmoid(reg_selected[:, 0, :])
    dy = torch.sigmoid(reg_selected[:, 1, :])
    
    # 尺寸解码
    # max=5.0 -> exp(5) ≈ 148 grid units. 
    # 如果 voxel=0.2m, 148 * 0.2 = 30m. 对于火车/障碍物足够了。
    w = torch.exp(torch.clamp(reg_selected[:, 3, :], min=-5.0, max=5.0))
    l = torch.exp(torch.clamp(reg_selected[:, 4, :], min=-5.0, max=5.0))
    
    # 【新增】旋转角度解码
    sin_rot = reg_selected[:, 6, :]
    cos_rot = reg_selected[:, 7, :]
    rot = torch.atan2(sin_rot, cos_rot) # 得到弧度值值
    
    # 4. 计算最终网格坐标
    final_x = topk_xs.float() + dx
    final_y = topk_ys.float() + dy
    
    # 组装结果: [x, y, w, l, rot, score, class] (共7个维度)
    det_boxes = torch.stack([
        final_x, final_y, w, l, rot, topk_scores, topk_classes.float()
    ], dim=2)
    
    return det_boxes

def bev_nms(boxes: torch.Tensor, iou_threshold: float = 0.3) -> torch.Tensor:
    """
    注意：现在的 boxes 维度是 [K, 7]
    [x, y, w, l, rot, score, class]
    """
    if boxes.shape[0] == 0:
        return torch.tensor([], dtype=torch.long, device=boxes.device)
    
    # 简单实现：使用 AABB (轴向外接矩形) 进行 NMS
    # 复杂实现可使用旋转 NMS，但铁路场景 AABB 通常足够
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    torch_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    scores = boxes[:, 5] # 第 5 位是 score
    
    keep = torchvision.ops.nms(torch_boxes, scores, iou_threshold)
    return keep