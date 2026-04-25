import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

from dataset import get_dataloaders
from model import AdaptiveInstrumentClassifier

# 超参数
DATA_PATH = r"D:\program_project\MUSE\data\IRMAS-TrainingData"
LOG_DIR = "model/log"
MODEL_DIR = "model"
EPOCHS = 50 # 增加迭代轮数以收敛高强度扩增数据
BATCH_SIZE = 64
LEARNING_RATE = 1e-4 # 使用转移学习（微调）时，必须将学习率调小 (5e-4 -> 1e-4) 以免毁掉预训练特征
WEIGHT_DECAY = 1e-3 # 加进 L2 正则化以制裁过拟合
NUM_CLASSES = 11

def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 获取数据
    train_loader, val_loader = get_dataloaders(DATA_PATH, batch_size=BATCH_SIZE, num_workers=4)

    # 实例化模型
    # 注意：这里的 config 必须与模型 __init__ 的参一致，这是自适应机制的核心
    model_config = {'num_classes': NUM_CLASSES, 'in_channels': 3}
    model = AdaptiveInstrumentClassifier(**model_config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    # 更换优化器策略: 换成带动量的 SGD 及 CosineAnnealingLR
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_epoch = 0
    best_acc = 0.0

    # 日志准备
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{timestamp}.log")
    
    # 尝试断点续训
    checkpoint_path = os.path.join(MODEL_DIR, "checkpoint_latest.pth")
    if os.path.exists(checkpoint_path):
        print(f"=> Loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        print(f"=> Loaded checkpoint (epoch {start_epoch}, best_acc {best_acc:.4f})")

    # 打开日志记录
    with open(log_file, "a") as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_Acc\n")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, targets in pbar_val:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}: Train Loss CPU: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 调度器自动调整
        scheduler.step()

        # 记录日志
        with open(log_file, "a") as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{val_acc:.4f}\n")

        # 随时保存最新 checkpoint（断点续训用）
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer_state': optimizer.state_dict(),
            'model_name': 'AdaptiveInstrumentClassifier',
            'config': model_config
        }, checkpoint_path)

        # 随时保存 best_model.pth (适配推理阶段的自适应加载)
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"--> New best accuracy ({best_acc:.4f}), saving best_model.pth!")
            best_model_path = os.path.join(MODEL_DIR, "best_model.pth")
            torch.save({
                'state_dict': model.state_dict(),
                'model_name': 'AdaptiveInstrumentClassifier',
                'config': model_config
            }, best_model_path)

if __name__ == "__main__":
    main()
