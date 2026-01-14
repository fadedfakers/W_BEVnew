import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from configs.config import BEVConfig as Config
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.logger import setup_logger

def train():
    """
    Main training loop.
    Supports Mixed Precision (AMP) and Multi-Task Loss computation.
    """
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logger('./work_dirs')
    
    # 2. Data
    dataset = BEVMultiTaskDataset(data_root='/root/autodl-tmp/FOD/data/1_calibration_1.2', split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    
    # 3. Model & Optimizer
    model = WBEVFusionNet(Config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()
    
    # 4. Loop
    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        for i, batch in enumerate(dataloader):
            with autocast():
                # Forward pass
                outputs = model(batch['image'], batch['points'], batch['calib'])
                
                # Multi-Task Loss = DetLoss + SegLoss
                # det_loss = calculate_det_loss(outputs, batch)
                # seg_loss = calculate_seg_loss(outputs, batch)
                # total_loss = det_loss + seg_loss
                pass
            
            # Backward pass
            # scaler.scale(total_loss).backward()
            # scaler.step(optimizer)
            # scaler.update()
            
            # logger.info(f"Epoch {epoch} | Step {i} | Loss: ...")

if __name__ == "__main__":
    train()
