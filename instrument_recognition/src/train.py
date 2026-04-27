# train.py
import os
import glob
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from src.config import config
from src.data import get_dataloaders
from src.model import SimplifiedAdvancedClassifier

def train():
    device = config.DEVICE
    train_loader, val_loader, classes = get_dataloaders(val_split=0.2)
    
    model_version = config.MODEL_VERSION
    model = SimplifiedAdvancedClassifier(num_classes=len(classes)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型版本: {model_version}")
    print(f"模型总参数量: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-4) # Re-add weight decay for VER1.7
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
    with open(log_file_path, "w") as f:
        f.write(f"Model Version: {model_version}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Resumed from Epoch: {start_epoch}\n")
        f.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\n")
    
    for epoch in range(start_epoch, config.EPOCHS):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # training loop with tqdm
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            current_loss = train_loss / (pbar.n + 1)
            current_acc = 100. * train_correct / train_total
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.2f}%"})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total

        # Eval
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total
        
        # Step scheduler
        scheduler.step()
        
        acc_percent = 100. * avg_val_acc
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Train Acc: {100.*avg_train_acc:.2f}% | Val Loss: {avg_val_loss:.4f}, Val Acc: {acc_percent:.2f}% | LR: {scheduler.get_last_lr()[0]:.6f}")
        
        with open(log_file_path, "a") as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_train_acc:.4f}\t{avg_val_loss:.4f}\t{avg_val_acc:.4f}\n")
        
        if acc_percent > best_acc:
            best_acc = acc_percent
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

if __name__ == "__main__":
    train()