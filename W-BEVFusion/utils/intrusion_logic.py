import torch
import numpy as np
from typing import List, Dict
import cv2

def check_intrusion(det_boxes: torch.Tensor, 
                    rail_mask: torch.Tensor, 
                    threshold_red: float = 0.3) -> List[Dict]:
    """
    检查检测框（支持旋转）与轨道掩膜的重叠情况。
    """
    alerts = []
    H, W = rail_mask.shape
    # 将轨道掩膜转为 numpy 方便 cv2 处理
    rail_mask_np = rail_mask.cpu().numpy().astype(np.uint8)
    
    # 转换为 numpy 提高循环内的处理速度
    boxes_np = det_boxes.cpu().numpy()

    for i in range(boxes_np.shape[0]):
        box = boxes_np[i]
        # box 格式: [x, y, w, l, rot, score, class]
        cx, cy, w, l, rot, score = box[0], box[1], box[2], box[3], box[4], box[5]
        
        if score < 0.1: continue

        # 1. 计算旋转矩形的四个顶点索引
        # 定义矩形框的四个角点（中心坐标系）
        corners = np.array([
            [-w/2, -l/2], [w/2, -l/2], [w/2, l/2], [-w/2, l/2]
        ])
        
        # 旋转矩阵
        res = np.array([[np.cos(rot), -np.sin(rot)], 
                        [np.sin(rot),  np.cos(rot)]])
        
        # 变换到图像网格坐标系
        rotated_corners = (corners @ res.T) + np.array([cx, cy])
        pts = rotated_corners.astype(np.int32)
        
        # 2. 生成该物体的精确 Footprint Mask
        obj_mask = np.zeros((H, W), dtype=np.uint8)
        cv2.fillPoly(obj_mask, [pts], 1)
        
        # 3. 计算重叠区域 (逻辑与操作)
        overlap = cv2.bitwise_and(obj_mask, rail_mask_np)
        
        num_rail_pixels = np.sum(overlap)
        total_box_pixels = np.sum(obj_mask)
        
        ratio = num_rail_pixels / total_box_pixels if total_box_pixels > 0 else 0
        
        # 4. 判定级别
        alert_level = "NONE"
        if ratio > threshold_red:
            alert_level = "RED"
        elif ratio > 0.01: # 只要有 1% 重叠即为黄色警告
            alert_level = "YELLOW"
            
        if alert_level != "NONE":
            # 获取外接矩形用于可视化
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            alerts.append({
                'box_id': i,
                'class': int(box[6]),
                'score': float(score),
                'ratio': float(ratio),
                'alert': alert_level,
                'bbox_grid': [int(x_min), int(y_min), int(x_max), int(y_max)],
                'poly_pts': pts.tolist() # 保存旋转多边形顶点
            })
            
    return alerts