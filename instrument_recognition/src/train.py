# train.py
import os
import json
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from tqdm import tqdm
from src.config import config
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import get_dataloaders
from src.model import TransformerClassifier


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced multi-label classification.
    FL = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t = p for y=1, 1-p for y=0.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # tensor of shape [num_classes] or None
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: [B, C], targets: [B, C] binary
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)  # p if y=1 else 1-p
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_weight * bce_loss
        else:
            loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def train():
    device = config.DEVICE
    train_loader, val_loader, classes, train_dist, val_dist = get_dataloaders(val_split=0.2)

    model_version = config.MODEL_VERSION
    model = TransformerClassifier(num_classes=len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型版本: {model_version}")
    print(f"模型总参数量: {total_params:,}")

    # Compute class weights for Focal Loss (inverse frequency, normalized to [0,1])
    class_counts = np.array([train_dist[c] for c in classes], dtype=np.float64)
    total = class_counts.sum()
    # Inverse frequency: rare classes get higher weight
    inv_freq = total / (len(classes) * class_counts)
    # Normalize to (0, 1) — alpha controls weight of positive samples
    alpha_weights = inv_freq / inv_freq.max()
    alpha_weights = np.clip(alpha_weights, 0.1, 0.9)
    alpha_tensor = torch.tensor(alpha_weights, dtype=torch.float32).to(device)
    print(f"Focal Loss alpha weights: {dict(zip(classes, alpha_weights.round(2)))}")

    criterion = FocalLoss(gamma=2.0, alpha=alpha_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    start_epoch = 0
    best_acc = 0.0
    checkpoint_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    latest_checkpoint_path = os.path.join(config.MODEL_DIR, "checkpoint_latest.pth")

    if os.path.exists(latest_checkpoint_path):
        try:
            ckpt = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
            if ckpt.get('version') == model_version:
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                start_epoch = ckpt['epoch'] + 1
                best_acc = ckpt.get('best_acc', 0.0)
                print(f"成功从最新的 Checkpoint 恢复！回到 Epoch {start_epoch}, 历史最佳准确率: {best_acc:.2f}%")
            else:
                print(f"Checkpoint 版本不匹配 (上次版本: {ckpt.get('version')}, 当前: {model_version})，将重新开始训练。")
        except Exception as e:
            print(f"加载 Checkpoint 失败，将重新开始训练。错误信息: {e}")

    if start_epoch >= config.EPOCHS:
        print(f"\n============================================\n"
              f"模型已经达到配置的最大 Epoch 数 ({config.EPOCHS})，无需继续训练！\n"
              f"如需继续训练该模型，请前往 src/config.py 调大 EPOCHS 参数。\n"
              f"============================================\n")
        return

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file_path = os.path.join(config.LOG_DIR, f"{timestamp}.log")
    with open(log_file_path, "w", encoding="utf-8") as f:
        f.write(f"Model Version: {model_version}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Resumed from Epoch: {start_epoch}\n")
        f.write(f"Train Dataset Breakdown: {train_dist}\n")
        f.write(f"Val Dataset Breakdown: {val_dist}\n")
        f.write("-" * 60 + "\n")
        f.write("Epoch\tTrain_Loss\tTrain_F1\tVal_Loss\tVal_F1\n")
    
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss = 0
        train_tp = 0
        train_fp = 0
        train_fn = 0
        
        # training loop with tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")

        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Mixup Augmentation
            alpha = 0.2
            if random.random() > 0.5:
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs.size(0)).to(device)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
                targets_a, targets_b = targets.float(), targets[index].float()
                
                optimizer.zero_grad()
                outputs = model(mixed_inputs)
                loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                
                predicted = (outputs > 0.0).float()
                mixed_targets = (lam * targets_a + (1 - lam) * targets_b > 0.5).float()
                
                train_tp += ((predicted == 1) & (mixed_targets == 1)).sum().item()
                train_fp += ((predicted == 1) & (mixed_targets == 0)).sum().item()
                train_fn += ((predicted == 0) & (mixed_targets == 1)).sum().item()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())
                
                predicted = (outputs > 0.0).float()
                
                train_tp += ((predicted == 1) & (targets == 1)).sum().item()
                train_fp += ((predicted == 1) & (targets == 0)).sum().item()
                train_fn += ((predicted == 0) & (targets == 1)).sum().item()
                
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            current_loss = train_loss / (pbar.n + 1)
            # 防止除以 0
            current_f1 = (2 * train_tp) / (2 * train_tp + train_fp + train_fn + 1e-8)
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'F1': f"{current_f1*100:.2f}%"})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_f1 = (2 * train_tp) / (2 * train_tp + train_fp + train_fn + 1e-8)

        # Eval
        model.eval()
        val_loss = 0
        val_tp = 0
        val_fp = 0
        val_fn = 0
        # 逐类统计，用于发现具体哪个乐器检测薄弱
        val_tp_per_class = torch.zeros(len(classes), device=device)
        val_fp_per_class = torch.zeros(len(classes), device=device)
        val_fn_per_class = torch.zeros(len(classes), device=device)

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.float())

                val_loss += loss.item()
                predicted = (outputs > 0.0).float()

                val_tp += ((predicted == 1) & (targets == 1)).sum().item()
                val_fp += ((predicted == 1) & (targets == 0)).sum().item()
                val_fn += ((predicted == 0) & (targets == 1)).sum().item()

                for c in range(len(classes)):
                    val_tp_per_class[c] += ((predicted[:, c] == 1) & (targets[:, c] == 1)).sum()
                    val_fp_per_class[c] += ((predicted[:, c] == 1) & (targets[:, c] == 0)).sum()
                    val_fn_per_class[c] += ((predicted[:, c] == 0) & (targets[:, c] == 1)).sum()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_f1 = (2 * val_tp) / (2 * val_tp + val_fp + val_fn + 1e-8)

        # 逐类 F1
        per_class_f1 = {}
        for c, cls_name in enumerate(classes):
            tp = val_tp_per_class[c].item()
            fp = val_fp_per_class[c].item()
            fn = val_fn_per_class[c].item()
            f1_c = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            per_class_f1[cls_name] = round(f1_c, 4)
        
        # Step scheduler
        scheduler.step()
        
        f1_percent = 100. * avg_val_f1
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train F1: {100.*avg_train_f1:.2f}% | Val Loss: {avg_val_loss:.4f}, Val F1: {f1_percent:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"  Per-Class F1: {per_class_f1}")
        
        with open(log_file_path, "a") as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_train_f1:.4f}\t{avg_val_loss:.4f}\t{avg_val_f1:.4f}\t{per_class_f1}\n")
        
        if f1_percent > best_acc:
            best_acc = f1_percent
            torch.save({
                'version': model_version,
                'model_state_dict': model.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }, checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")
            
        torch.save({
            'epoch': epoch,
            'version': model_version,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_acc': best_acc
        }, latest_checkpoint_path)

    # 训练完成后计算并保存逐类最优判定阈值
    print("\n正在计算逐类最优判定阈值...")
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu()
            all_probs.append(probs)
            all_targets.append(targets)

    all_probs = torch.cat(all_probs, dim=0)      # [N, C]
    all_targets = torch.cat(all_targets, dim=0)  # [N, C]

    per_class_thresholds = {}
    for c, cls_name in enumerate(classes):
        best_f1 = 0.0
        best_thresh = 0.5
        probs_c = all_probs[:, c].numpy()
        targets_c = all_targets[:, c].numpy()
        for thresh in np.arange(0.2, 0.85, 0.05):
            preds = (probs_c >= thresh).astype(float)
            tp = ((preds == 1) & (targets_c == 1)).sum()
            fp = ((preds == 1) & (targets_c == 0)).sum()
            fn = ((preds == 0) & (targets_c == 1)).sum()
            f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        per_class_thresholds[cls_name] = round(float(best_thresh), 2)
        print(f"  {cls_name}: 最佳阈值={best_thresh:.2f}, F1={best_f1:.4f}")

    threshold_path = os.path.join(config.MODEL_DIR, "class_thresholds.json")
    with open(threshold_path, "w") as f:
        json.dump(per_class_thresholds, f, indent=2)
    print(f"逐类阈值已保存至: {threshold_path}")


if __name__ == "__main__":
    train()