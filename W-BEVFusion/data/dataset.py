import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import json
import os
import open3d as o3d
import raillabel
from configs.config import BEVConfig

class BEVMultiTaskDataset(Dataset):
    def __init__(self, data_root, split='train'):
        self.data_root = data_root
        # 实例化 Config，方便后续调用
        self.cfg = BEVConfig() if callable(BEVConfig) else BEVConfig
        
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"❌ Data root not found: {data_root}")
            
        # --- 场景扫描逻辑 ---
        if os.path.exists(os.path.join(data_root, "lidar")):
            print(f"⚠️ Warning: data_root seems to be a scene folder. Using it as a single scene.")
            self.scenes = [os.path.basename(data_root)]
            self.data_root = os.path.dirname(data_root)
        else:
            all_scenes = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
            all_scenes.sort()
            split_idx = int(len(all_scenes) * 0.8)
            self.scenes = all_scenes[:split_idx] if split == 'train' else all_scenes[split_idx:]
            
        self.samples = []
        self._collect_samples()
        print(f"[{split.upper()}] Loaded {len(self.samples)} valid samples.")

    def __len__(self):
        return len(self.samples)

    def _collect_samples(self):
        print(f"Scanning {len(self.scenes)} scenes...")
        for scene_id in self.scenes:
            scene_dir = os.path.join(self.data_root, scene_id)
            
            # 1. 查找 JSON
            # 优先查找标准命名 {scene_id}_labels.json
            json_name = f"{scene_id}_labels.json"
            json_path = os.path.join(scene_dir, json_name)
            
            if not os.path.exists(json_path):
                # 模糊查找
                candidates = [f for f in os.listdir(scene_dir) if f.endswith('_labels.json')]
                if candidates:
                    json_path = os.path.join(scene_dir, candidates[0])
                else:
                    # 如果没有 label 文件，跳过该场景
                    continue
            
            # 2. 加载 raillabel 数据
            try:
                scene = raillabel.load(json_path)
            except Exception as e:
                print(f"  ❌ Error loading {json_path}: {e}")
                continue
            
            # 检查文件夹是否存在
            lidar_dir = os.path.join(scene_dir, "lidar")
            rgb_dir = os.path.join(scene_dir, "rgb_center") # 你的数据集中文件夹名为 rgb_center
            
            if not os.path.exists(lidar_dir) or not os.path.exists(rgb_dir):
                print(f"  ⚠️ Scene {scene_id} missing 'lidar' or 'rgb_center' folder. Skipped.")
                continue
                
            # 优化：预先读取文件夹下的所有文件，避免在循环中反复 IO
            all_pcd = sorted([f for f in os.listdir(lidar_dir) if f.endswith('.pcd')])
            # 支持 jpg 和 png
            all_img = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # 3. 匹配帧 (修复核心逻辑)
            # 你的文件名格式为 "000_timestamp.png"，而 raillabel 的 frame_id 通常是 0, 1, 2...
            for frame_id, frame in scene.frames.items():
                try:
                    fid_int = int(frame_id)
                    # 构造前缀: 0 -> "000_"
                    prefix = f"{fid_int:03d}_"
                except ValueError:
                    prefix = str(frame_id)
                
                # 在文件列表中查找匹配前缀的文件
                pcd_file = next((f for f in all_pcd if f.startswith(prefix)), None)
                img_file = next((f for f in all_img if f.startswith(prefix)), None)
                
                if pcd_file and img_file:
                    self.samples.append({
                        'pcd_path': os.path.join(lidar_dir, pcd_file),
                        'img_path': os.path.join(rgb_dir, img_file),
                        'frame': frame,
                        'scene': scene,
                        'scene_id': scene_id,
                        'frame_id': frame_id,
                        'calib_path': os.path.join(scene_dir, "calibration.txt") # 记录标定文件路径
                    })

    def _world_to_grid(self, x, y):
        # 将世界坐标 (meters) 转换为 BEV Grid 坐标 (pixels)
        gx = int((x - self.cfg.X_RANGE[0]) / self.cfg.VOXEL_SIZE)
        gy = int((y - self.cfg.Y_RANGE[0]) / self.cfg.VOXEL_SIZE)
        gx = max(0, min(gx, self.cfg.GRID_W - 1))
        gy = max(0, min(gy, self.cfg.GRID_H - 1))
        return gx, gy

    def _load_calib_data(self, item):
        """
        加载标定数据的占位函数。
        如果后续模型需要内参/外参，你需要在这里解析 calibration.txt。
        目前为了防崩，返回单位矩阵。
        """
        # TODO: 解析 item['calib_path'] 获取真实的 K, R, T
        # 假设 calibration.txt 包含相机内参 K 和雷达-相机外参 T
        
        # 默认返回单位矩阵，防止 KeyErrors
        return {
            "camera_intrinsics": torch.eye(3), # K
            "camera2lidar": torch.eye(4),      # T
            "lidar2camera": torch.eye(4)       # inv(T)
        }

    def __getitem__(self, idx):
        item = self.samples[idx]
        
        # --- 1. Load LiDAR ---
        try:
            pcd = o3d.io.read_point_cloud(item['pcd_path'])
            points = np.asarray(pcd.points)
        except Exception:
            points = np.zeros((1, 3), dtype=np.float32)

        # 裁剪点云范围 (BEV Range)
        mask = (points[:, 0] >= self.cfg.X_RANGE[0]) & (points[:, 0] < self.cfg.X_RANGE[1]) & \
               (points[:, 1] >= self.cfg.Y_RANGE[0]) & (points[:, 1] < self.cfg.Y_RANGE[1]) & \
               (points[:, 2] >= self.cfg.Z_RANGE[0]) & (points[:, 2] < self.cfg.Z_RANGE[1])
        points = points[mask]
        
        # 处理空点云或补齐 Intensity
        if len(points) == 0:
            points = np.zeros((1, 4), dtype=np.float32)
        elif points.shape[1] == 3:
            points = np.hstack([points, np.ones((points.shape[0], 1))]).astype(np.float32)
            
        # --- 2. Load Image (修复几何畸变) ---
        img_path = item['img_path']
        img = cv2.imread(img_path)
        
        # 错误检查：如果读不到图，直接报错，不给假数据
        if img is None:
            raise FileNotFoundError(f"❌ Critical Error: Image not found or corrupted: {img_path}")
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # [FIX] 使用 Config 中定义的尺寸，而不是硬编码的 (640, 480)
        # 这确保了如果你的 Config 是 1280x720 (16:9)，图片就不会被压扁
        target_w, target_h = self.cfg.IMG_SIZE
        img = cv2.resize(img, (target_w, target_h))
            
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        # --- 3. Load Calibration (新功能) ---
        calib_dict = self._load_calib_data(item)

        # --- 4. Generate Targets ---
        rail_mask = np.zeros((self.cfg.GRID_H, self.cfg.GRID_W), dtype=np.uint8)
        boxes = []
        labels = []
        
        frame = item['frame']
        scene = item['scene']
        
        for ann_id, ann in frame.annotations.items():
            if not hasattr(ann, 'object_id'): continue
            if ann.object_id not in scene.objects: continue
            
            obj = scene.objects[ann.object_id]
            obj_type = obj.type.lower()
            
            # === 分支 A: 铁轨语义分割 (Rails) ===
            if 'track' in obj_type or 'rail' in obj_type:
                if isinstance(ann, raillabel.format.Poly3d):
                    pts_grid = []
                    for pt in ann.points:
                        # 兼容处理 raillabel 版本差异
                        if hasattr(pt, 'x'):
                            px, py = float(pt.x), float(pt.y)
                        else:
                            px, py = float(pt[0]), float(pt[1])
                            
                        gx, gy = self._world_to_grid(px, py)
                        pts_grid.append([gx, gy])
                    
                    if len(pts_grid) > 1:
                        pts_grid = np.array(pts_grid, dtype=np.int32)
                        # 注意：OpenCV 画图是在 numpy 数组上，坐标系也是 (x, y)
                        cv2.polylines(rail_mask, [pts_grid], isClosed=False, color=1, thickness=self.cfg.RAIL_MASK_THICKNESS)

            # === 分支 B: 障碍物检测 (Obstacles) ===
            OBSTACLE_WHITELIST = ['person', 'vehicle', 'car', 'truck', 'van', 'bicycle', 'motorcycle', 'animal', 'crowd', 'obstacle', 'debris', 'rock']
            
            if any(x in obj_type for x in OBSTACLE_WHITELIST):
                if hasattr(ann, 'pos') and hasattr(ann, 'size'):
                    cx = ann.pos.x if hasattr(ann.pos, 'x') else ann.pos[0]
                    cy = ann.pos.y if hasattr(ann.pos, 'y') else ann.pos[1]
                    # 注意：通常 size 是 [dx, dy, dz] 或者 [length, width, height]
                    # 这里的解析取决于 raillabel 的定义，假设 x 为长，y 为宽
                    dx = ann.size.x if hasattr(ann.size, 'x') else ann.size[0]
                    dy = ann.size.y if hasattr(ann.size, 'y') else ann.size[1]
                    
                    gx, gy = self._world_to_grid(float(cx), float(cy))
                    gw = int(float(dx) / self.cfg.VOXEL_SIZE)
                    gl = int(float(dy) / self.cfg.VOXEL_SIZE)
                    
                    if gw < 1 or gl < 1: continue 
                    
                    if 0 <= gx < self.cfg.GRID_W and 0 <= gy < self.cfg.GRID_H:
                        boxes.append([gx, gy, gw, gl])
                        labels.append(1)

        target_dict = {
            'masks': torch.from_numpy(rail_mask).long(),
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
            # 将标定数据也放入 target
            **calib_dict
        }
        
        return img_tensor, torch.from_numpy(points), target_dict

    @staticmethod
    def collate_fn(batch):
        images = torch.stack([b[0] for b in batch])
        points = [b[1] for b in batch]
        targets = [b[2] for b in batch]
        return images, points, targets