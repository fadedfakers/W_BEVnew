import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader

# 路径 hack
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.box_ops import decode_boxes, bev_nms

def calculate_iou(pred_mask, gt_mask):
    """
    计算二值化后的 IoU.
    pred_mask: (H, W) float [0, 1]
    gt_mask: (H, W) int {0, 1}
    """
    pred_bin = (pred_mask > 0.5).astype(np.uint8)
    gt_bin = (gt_mask > 0.5).astype(np.uint8)
    
    intersection = (pred_bin & gt_bin).sum()
    union = (pred_bin | gt_bin).sum()
    
    # 特殊情况处理
    if union == 0:
        # 如果 GT 为空，预测也为空 -> 1.0
        # 如果 GT 为空，预测不为空 -> 0.0
        return 1.0 if pred_bin.sum() == 0 else 0.0
        
    return intersection / (union + 1e-6)

@torch.no_grad()
def run_full_evaluation(checkpoint_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing model on {device}...")
    model = WBEVFusionNet(cfg).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint not found at {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    sd = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    model.load_state_dict(sd)
    model.eval()
    print(f"✅ Loaded checkpoint: {checkpoint_path}")

    # 使用 val split
    dataset = BEVMultiTaskDataset(data_root=data_root, split='val')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=dataset.collate_fn)

    ious = []
    all_scores = []
    all_matches = [] # 1 for TP, 0 for FP
    total_gts = 0
    nan_count = 0

    print(f"🏁 Validating {len(dataset)} samples...")
    
    for i, (images, points, targets) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        points_list = [p.to(device) for p in points]
        
        outputs = model(images, points_list)
        
        # --- 1. Segmentation Eval ---
        mask_logit = outputs['mask_pred'][0, 0]
        if torch.isnan(mask_logit).any():
            nan_count += 1
            continue

        pred_mask = torch.sigmoid(mask_logit).cpu().numpy()
        gt_mask = targets[0]['masks'].numpy()
        ious.append(calculate_iou(pred_mask, gt_mask))

        # --- 2. Detection Eval ---
        # Decode boxes
        det_boxes_batch = decode_boxes(outputs, K=100, threshold=0.1)
        det_boxes = det_boxes_batch[0]
        
        # NMS
        keep = bev_nms(det_boxes, iou_threshold=0.3)
        det_boxes = det_boxes[keep].cpu().numpy()

        # Get GT boxes
        gt_boxes = targets[0]['boxes'].numpy() # [x, y, w, l] in Grid
        total_gts += len(gt_boxes)

        # Matching Logic (Greedy)
        matched_gt_indices = set()
        
        # 对预测框按分数排序 (虽然 decode_boxes 已经排过了，保险起见)
        # det_boxes: [x, y, w, l, rot, score, class]
        if len(det_boxes) > 0:
            det_boxes = det_boxes[np.argsort(-det_boxes[:, 5])]

        for det in det_boxes:
            det_x, det_y = det[0], det[1]
            score = det[5]
            
            all_scores.append(score)
            
            is_tp = False
            best_dist = float('inf')
            best_gt_idx = -1
            
            # 寻找最近的未匹配 GT
            for g_idx, gt in enumerate(gt_boxes):
                if g_idx in matched_gt_indices:
                    continue
                
                gt_x, gt_y = gt[0], gt[1]
                # 计算物理距离 (米)
                dist_m = np.sqrt((det_x - gt_x)**2 + (det_y - gt_y)**2) * cfg.VOXEL_SIZE
                
                # 距离阈值: 2.0米 (对于障碍物检测比较合理)
                if dist_m < 2.0 and dist_m < best_dist:
                    best_dist = dist_m
                    best_gt_idx = g_idx
            
            if best_gt_idx != -1:
                is_tp = True
                matched_gt_indices.add(best_gt_idx)
            
            all_matches.append(1 if is_tp else 0)

    # --- Metrics Calculation ---
    
    # mIoU
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
    
    # AP (Average Precision)
    all_scores = np.array(all_scores)
    all_matches = np.array(all_matches)
    ap = 0.0
    
    if len(all_scores) > 0 and total_gts > 0:
        # Sort by score high -> low
        sorted_indices = np.argsort(-all_scores)
        all_matches = all_matches[sorted_indices]
        
        # Compute Precision/Recall Curve
        tp_cumsum = np.cumsum(all_matches)
        fp_cumsum = np.cumsum(1 - all_matches)
        
        recalls = tp_cumsum / total_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        
        # Smooth P-R Curve (VOC style)
        precisions = np.maximum.accumulate(precisions[::-1])[::-1]
        
        # AUC (Area Under Curve)
        # 处理 x 轴 (recall) 重复点
        ap = np.trapz(precisions, recalls) 
        # 防止负值 (trapz 可能因为 x 轴顺序问题出负，虽然这里 recalls 单调增)
        ap = max(0.0, ap)

    print("\n" + "="*40)
    print(f"📊 EVALUATION REPORT")
    print(f"  - Rail mIoU:     {mean_iou*100:.2f} %")
    print(f"  - Obstacle AP:   {ap*100:.2f} %")
    print(f"  - NaN Samples:   {nan_count}")
    print(f"  - Valid Samples: {len(ious)}")
    print("="*40)

if __name__ == "__main__":
    # 请修改为实际路径
    CKPT = "checkpoints/20260113_XXXX/model_best.pth"
    DATA = "/root/autodl-tmp/FOD/data"
    
    # 自动搜索最新的
    if not os.path.exists(CKPT):
        import glob
        list_of_files = glob.glob('checkpoints/*/*.pth') 
        if list_of_files:
            CKPT = max(list_of_files, key=os.path.getctime)
            print(f"🔎 Auto-selected checkpoint: {CKPT}")

    run_full_evaluation(CKPT, DATA)