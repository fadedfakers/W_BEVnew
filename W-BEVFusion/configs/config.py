import numpy as np

class BEVConfig:
    """
    Configuration class for W-BEVFusion Data Pipeline.
    Refined for better small object detection and coordinate clarity.
    """
    # ==========================================
    # 1. World Bounds (LiDAR Coordinate System)
    # ==========================================
    # 建议：根据 LiDAR 安装高度调整 Z_RANGE
    # X: Forward, Y: Left/Right, Z: Up/Down
    X_RANGE = (0.0, 102.4)     # 102.4 / 0.4 = 256 (32的倍数，OK)
    Y_RANGE = (-25.6, 25.6)    # 51.2 / 0.4 = 128 (32的倍数，完美!)
    Z_RANGE = (-5.0, 3.0)    # [建议] 扩大 Z 轴范围，防止地面或高处物体被切除

    # ==========================================
    # 2. Voxel/Grid Settings (Crucial for Accuracy)
    # ==========================================
    # [优化] 从 0.4 改为 0.2 或 0.1。
    # 0.4m 对于检测石头/行人来说太粗糙了。
    VOXEL_SIZE = 0.2  
    
    # Calculated Grid Size
    # New Grid Size with 0.2m voxel:
    # W (Forward): 102.4 / 0.2 = 512
    # H (Lateral): 40.96 / 0.2 = 204.8 -> 205 (or 204)
    # 注意：这里为了方便矩阵运算，通常确保 Grid 是 16 或 32 的倍数
    GRID_W = int((X_RANGE[1] - X_RANGE[0]) / VOXEL_SIZE) # 512
    GRID_H = int((Y_RANGE[1] - Y_RANGE[0]) / VOXEL_SIZE) # 204

    # ==========================================
    # 3. Image Settings
    # ==========================================
    # 保持 16:9 比例，避免投影拉伸
    # 如果显存不够，可以改为 (960, 540) 或 (800, 450)
    IMG_SIZE = (1280, 720) 
    
    # [新增] 图像归一化参数 (ImageNet Standard)
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD  = [0.229, 0.224, 0.225]

    # ==========================================
    # 4. Model & Training Settings
    # ==========================================
    BATCH_SIZE = 4       # 根据显存调整，太小会导致 BatchNorm 不稳
    LEARNING_RATE = 2e-4 # 稍微调大一点点，或者使用 Warmup
    NUM_EPOCHS = 50
    
    NUM_CLASSES = 2      # 0: Background, 1: Obstacle
    
    # [优化] 检测框编码
    # 通常 CenterPoint 用 9 或 10 码 (加 velocity)，
    # 但如果是单帧检测，8 码够了: [x, y, z, w, l, h, sin, cos]
    BOX_CODE_SIZE = 8     

    # ==========================================
    # 5. Visualization / Post-processing
    # ==========================================
    RAIL_MASK_THICKNESS = 2 # 像素变多了，线可以细一点
    
    # [新增] 阈值设置
    CONF_THRESHOLD = 0.5    # 置信度阈值
    IOU_THRESHOLD = 0.1     # NMS 阈值

    def __init__(self):
        # 确保这个检查通过
        if self.GRID_W % 32 != 0 or self.GRID_H % 32 != 0:
            raise ValueError(f"❌ Grid Size ({self.GRID_W}, {self.GRID_H}) 必须能被 32 整除！请调整 Range 或 Voxel Size。")