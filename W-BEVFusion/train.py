import os
import torch
import time
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# 确保这些模块路径正确
from configs.config import BEVConfig
from data.dataset import BEVMultiTaskDataset
from models.detector import WBEVFusionNet
from utils.losses import MultiTaskLoss

def parse_args():
    parser = argparse.ArgumentParser(description="Train W-BEVFusion")
    parser.add_argument('--data_root', type=str, default="/root/autodl-tmp/FOD/data", help="Path to dataset")
    parser.add_argument('--batch_size', type=int, default=BEVConfig.BATCH_SIZE)
    parser.add_argument('--epochs', type=int, default=BEVConfig.NUM_EPOCHS)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 动态生成保存路径
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"🚀 Training starting on {device}")
    print(f"📂 Data Root: {args.data_root}")
    print(f"💾 Checkpoints will be saved to: {save_dir}")

    # --- 1. Dataset & Loader ---
    try:
        train_dataset = BEVMultiTaskDataset(data_root=args.data_root, split='train')
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        collate_fn=BEVMultiTaskDataset.collate_fn, 
        drop_last=True,
        pin_memory=True # 【优化】加速 CPU 到 GPU 传输
    )

    # --- 2. Model, Optimizer, Scheduler ---
    model = WBEVFusionNet(config=BEVConfig).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=BEVConfig.LEARNING_RATE, weight_decay=0.01)
    
    # OneCycleLR 非常适合这种从零开始的训练
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=BEVConfig.LEARNING_RATE, 
        steps_per_epoch=len(train_loader), 
        epochs=args.epochs,
        pct_start=0.3 # 前 30% 时间热身
    )
    
    criterion = MultiTaskLoss()
    scaler = torch.amp.GradScaler('cuda')

    # --- 3. Training Loop ---
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        valid_batches = 0
        
        # Tqdm 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for images, points, targets in pbar:
            # Move to device
            images = images.to(device)
            points = [p.to(device) for p in points]
            # targets 中的 tensor 已经在 Dataset 或 Loss 内部移动了，或者在这里处理
            
            optimizer.zero_grad(set_to_none=True) # 【优化】稍微快一点
            
            # --- Forward & Loss (Mixed Precision) ---
            with torch.amp.autocast('cuda'):
                predictions = model(images, points)
                loss_dict = criterion(predictions, targets)
                loss = loss_dict['total_loss']
            
            # --- Backward ---
            if torch.isnan(loss):
                print("⚠️ Warning: NaN loss detected. Skipping batch.")
                continue

            scaler.scale(loss).backward()
            
            # Gradient Clipping (Unscale first)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # 1.0 可能太紧了，5.0 比较通用
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Logging
            curr_loss = loss.item()
            epoch_loss += curr_loss
            valid_batches += 1
            
            pbar.set_postfix({
                'Loss': f"{curr_loss:.3f}", 
                'Cls': f"{loss_dict['loss_cls'].item():.3f}",
                'Box': f"{loss_dict['loss_box'].item():.3f}",
                'Seg': f"{loss_dict['loss_seg'].item():.3f}",
                'LR': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
        # End of Epoch
        avg_loss = epoch_loss / valid_batches if valid_batches > 0 else 0
        print(f"Epoch {epoch+1} Done. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(save_dir, f"model_last.pth")
        torch.save(model.state_dict(), save_path)
        
        # Save Best
        if avg_loss < best_loss and valid_batches > 0:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "model_best.pth"))
            print(f"⭐ New best model saved (Loss: {best_loss:.4f})")
            
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_e{epoch+1}.pth"))

if __name__ == "__main__":
    main()