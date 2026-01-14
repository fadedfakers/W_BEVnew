import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, w_cls=10.0, w_box=2.0, w_seg=1.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.w_cls = w_cls # 分类难度大，权重给高点
        self.w_box = w_box
        self.w_seg = w_seg

    def _focal_loss(self, preds, targets):
        """
        preds: (B, 2, H, W) Logits
        targets: (B, H, W) 0 or 1
        """
        # 展平处理
        preds = preds.permute(0, 2, 3, 1).reshape(-1, 2)
        targets = targets.reshape(-1)
        
        # Cross Entropy 本身包含 LogSoftmax
        # reduction='none' 保留每个样本的 loss 以便乘 alpha
        ce_loss = F.cross_entropy(preds, targets, reduction='none')
        
        # 计算概率 pt
        pt = torch.exp(-ce_loss)
        
        # Focal term
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    def _dice_loss(self, preds, targets, smooth=1.0):
        # preds 是 Logits，需要 Sigmoid
        preds = torch.sigmoid(preds).reshape(-1)
        targets = targets.reshape(-1)
        
        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice

    def forward(self, predictions, targets):
        device = predictions['cls_pred'].device
        cls_pred = predictions['cls_pred'] # (B, 2, H, W)
        box_pred = predictions['box_pred'] # (B, 8, H, W)
        mask_pred = predictions['mask_pred'] # (B, 1, H, W)
        
        B, _, H, W = cls_pred.shape
        
        # 初始化目标
        target_cls = torch.zeros((B, H, W), dtype=torch.long, device=device)
        # 初始化 Box 目标，注意这里应该和 box_pred 通道数一致
        target_box = torch.zeros((B, 8, H, W), dtype=torch.float32, device=device)
        box_mask = torch.zeros((B, H, W), dtype=torch.bool, device=device)
        
        # 处理 Segmentation Target
        target_seg = torch.stack([t['masks'] for t in targets]).to(device)
        
        # 循环构建 Batch 内的 Detection Target
        for b in range(B):
            t_boxes = targets[b]['boxes']
            if len(t_boxes) > 0:
                t_boxes = t_boxes.to(device)
                
                # 1. 坐标映射到 Grid (确保不越界)
                gx = t_boxes[:, 0].long().clamp(0, W-1)
                gy = t_boxes[:, 1].long().clamp(0, H-1)
                
                # 2. 设置正样本中心 (Classification)
                target_cls[b, gy, gx] = 1 
                
                # 3. 设置回归目标 (Box Regression)
                # 假设 t_boxes 是 [gx, gy, gw, gl] (Grid单位)
                # 我们只回归 Log(w) 和 Log(l)，分别对应通道 3 和 4 (根据 Config BOX_CODE_SIZE 定义)
                # 修正重复计算 bug，统一 clamp
                gw = torch.clamp(t_boxes[:, 2], min=1e-4, max=200.0) 
                gl = torch.clamp(t_boxes[:, 3], min=1e-4, max=200.0)
                
                # 填入目标 Tensor
                target_box[b, 3, gy, gx] = torch.log(gw)
                target_box[b, 4, gy, gx] = torch.log(gl)
                
                # 记录哪些位置有 Box
                box_mask[b, gy, gx] = True

        # --- 计算 Losses ---
        
        # 1. Classification Loss (Focal)
        loss_cls = self._focal_loss(cls_pred, target_cls)
        
        # 2. Box Regression Loss (Smooth L1)
        # 只计算有物体的像素点
        if box_mask.sum() > 0:
            # 提取正样本预测值: (N_pos, 8)
            masked_pred = box_pred.permute(0,2,3,1)[box_mask]
            masked_target = target_box.permute(0,2,3,1)[box_mask]
            
            # 【重要】只计算我们设定了目标的通道 (3:5 即 w, l)
            # 其他通道(x,y,z,rot)暂时忽略，否则会强行回归到 0
            loss_box = F.smooth_l1_loss(masked_pred[:, 3:5], masked_target[:, 3:5], reduction='mean')
        else:
            loss_box = torch.tensor(0.0, device=device)
            
        # 3. Segmentation Loss (Dice + BCE)
        loss_seg_dice = self._dice_loss(mask_pred, target_seg.float())
        loss_seg_bce = F.binary_cross_entropy_with_logits(mask_pred.squeeze(1), target_seg.float())
        loss_seg = loss_seg_dice + loss_seg_bce
        
        # 总 Loss
        total_loss = (self.w_cls * loss_cls) + (self.w_box * loss_box) + (self.w_seg * loss_seg)
        
        return {
            'total_loss': total_loss, 
            'loss_cls': loss_cls, 
            'loss_box': loss_box, 
            'loss_seg': loss_seg, 
            'pos_count': box_mask.sum()
        }