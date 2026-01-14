import os
import sys

# 路径 hack
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from configs.config import BEVConfig as cfg
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.box_ops import decode_boxes, bev_nms
from utils.intrusion_logic import check_intrusion

def visualize_2x2(image, lidar_bev, pred_mask, det_boxes, alerts, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9)) # 宽屏布局
    
    # 1. RGB Image
    # 反归一化
    mean = np.array(cfg.IMG_MEAN).reshape(1, 1, 3)
    std = np.array(cfg.IMG_STD).reshape(1, 1, 3)
    image_display = (image * std + mean) * 255.0
    image_display = np.clip(image_display, 0, 255).astype(np.uint8)
    
    axes[0, 0].imshow(image_display)
    axes[0, 0].set_title("Input RGB Image")
    axes[0, 0].axis('off')
    
    # 2. LiDAR BEV Features
    # 取 Channel 的最大值，通常比均值更清晰地显示结构
    if lidar_bev.ndim == 3:
        lidar_img = lidar_bev.max(axis=0) 
    else:
        lidar_img = lidar_bev
    axes[0, 1].imshow(lidar_img, cmap='viridis', origin='upper') # 注意 origin
    axes[0, 1].set_title(f"LiDAR BEV Features (Max Pool)")
    axes[0, 1].axis('off')
    
    # 3. Predicted Rail Mask
    # 转为可视化的图 (H, W)
    axes[1, 0].imshow(pred_mask, cmap='gray', origin='upper')
    axes[1, 0].set_title("Predicted Rail Mask")
    axes[1, 0].axis('off')
    
    # 4. Safety Analysis Result (Canvas)
    # 创建画布，大小与 Grid 一致 (H, W, 3)
    H, W = pred_mask.shape
    result_bev = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 绘制铁轨 (蓝色)
    result_bev[pred_mask > 0.5] = [60, 60, 180] 
    
    # 绘制检测框和警报
    for alert in alerts:
        x1, y1, x2, y2 = alert['bbox_grid']
        
        # 颜色: BGR 格式 (OpenCV) -> 转 RGB 显示
        color = (255, 255, 0) # Yellow
        label = "WARN"
        if alert['alert'] == "RED":
            color = (255, 0, 0) # Red
            label = "STOP"
            
        # 画矩形 (x1, y1) 是左上角
        cv2.rectangle(result_bev, (x1, y1), (x2, y2), color, 1)
        
        # 标签
        font_scale = 0.5
        cv2.putText(result_bev, label, (x1, max(y1-2, 5)), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    axes[1, 1].imshow(result_bev, origin='upper')
    axes[1, 1].set_title(f"Safety Analysis (Grid: {W}x{H})")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✅ Saved visualization to {save_path}")

def evaluate(checkpoint_path, data_root):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Loading model from {checkpoint_path}...")
    
    model = WBEVFusionNet(cfg).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
    else:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    model.eval()
    
    try:
        dataset = BEVMultiTaskDataset(data_root=data_root, split='val')
    except Exception as e:
        print(f"❌ Dataset Error: {e}")
        return

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    
    print("📸 Starting visualization loop...")
    with torch.no_grad():
        for i, (images, points, targets) in enumerate(dataloader):
            if i >= 10: break # 只看前 10 张
            
            images = images.to(device)
            points_list = [p.to(device) for p in points]
            
            # Forward
            outputs = model(images, points_list)
            
            # Post-Process
            # K=50, threshold=0.2 (稍微严格一点)
            det_boxes_batch = decode_boxes(outputs, K=50, threshold=0.2) 
            det_boxes = det_boxes_batch[0] # 取 Batch 第一个
            
            # NMS
            keep = bev_nms(det_boxes, iou_threshold=0.1)
            det_boxes = det_boxes[keep]
            
            # Intrusion Logic
            # Mask Logit -> Sigmoid -> Binary
            rail_mask_logit = outputs['mask_pred'][0, 0]
            rail_mask = (torch.sigmoid(rail_mask_logit) > 0.5).float()
            
            alerts = check_intrusion(det_boxes, rail_mask)
            
            # Prepare Data for Plotting
            img_np = images[0].permute(1, 2, 0).cpu().numpy() # (H, W, 3)
            # 获取 LiDAR Feature 用于可视化
            # 注意：WBEVFusionNet 内部没有直接暴露 bev_map，你需要确保 detector.py 里 forward 返回了，或者在这里 hook
            # 为了简单，我们临时再次调用 backbone
            lidar_bev_map = model.lidar_backbone(points_list)[0].cpu().numpy()
            mask_np = rail_mask.cpu().numpy()
            
            save_path = f"vis_sample_{i:02d}.png"
            visualize_2x2(img_np, lidar_bev_map, mask_np, det_boxes.cpu().numpy(), alerts, save_path)

if __name__ == "__main__":
    # 自动查找最新的 checkpoint
    ckpt_dir = "checkpoints"
    if os.path.exists(ckpt_dir):
        folders = sorted([f for f in os.listdir(ckpt_dir) if f.startswith("20")])
        if folders:
            latest_folder = folders[-1]
            # 找该文件夹下最新的 pth
            pth_files = [f for f in os.listdir(os.path.join(ckpt_dir, latest_folder)) if f.endswith(".pth")]
            if pth_files:
                # 简单排序：model_e5.pth, model_best.pth... 
                # 这里假设我们要找最后的或者 best
                if "model_best.pth" in pth_files:
                    target_pth = "model_best.pth"
                else:
                    target_pth = sorted(pth_files)[-1]
                
                CHECKPOINT = os.path.join(ckpt_dir, latest_folder, target_pth)
            else:
                CHECKPOINT = "dummy.pth"
        else:
            CHECKPOINT = "dummy.pth"
    else:
        CHECKPOINT = "dummy.pth"

    DATA_ROOT = "/root/autodl-tmp/FOD/data"
    
    print(f"🔎 Auto-detected checkpoint: {CHECKPOINT}")
    evaluate(CHECKPOINT, DATA_ROOT)