import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime

from dataset import get_dataloaders
from model import AdaptiveInstrumentClassifier
from config import Config
from model_trainer import ModelTrainer

def main():
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 获取数据
    train_loader, val_loader = get_dataloaders(Config.DATA_PATH, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS)

    # 实例化模型
    model_config = {'num_classes': Config.NUM_CLASSES, 'in_channels': Config.IN_CHANNELS}
    model = AdaptiveInstrumentClassifier(**model_config)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.LEARNING_RATE, momentum=Config.MOMENTUM, weight_decay=Config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    trainer = ModelTrainer(model, optimizer, criterion, scheduler, device)

    # 日志准备
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(Config.LOG_DIR, f"{timestamp}.log")
    
    # 尝试断点续训
    checkpoint_path = os.path.join(Config.MODEL_DIR, "checkpoint_latest.pth")
    trainer.resume_from_checkpoint(checkpoint_path)

    # 打开日志记录
    with open(log_file, "a") as f:
        f.write("Epoch\tTrain_Loss\tVal_Loss\tVal_Acc\n")

    for epoch in range(trainer.start_epoch, Config.EPOCHS):
        avg_train_loss = trainer.train_epoch(train_loader, epoch)
        avg_val_loss, val_acc = trainer.val_epoch(val_loader, epoch)

        print(f"Epoch {epoch+1}: Train Loss CPU: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 调度器自动调整
        scheduler.step()

        # 记录日志
        with open(log_file, "a") as f:
            f.write(f"{epoch+1}\t{avg_train_loss:.4f}\t{avg_val_loss:.4f}\t{val_acc:.4f}\n")

        # 随时保存最新 checkpoint
        trainer.save_checkpoint(checkpoint_path, epoch + 1, model_config)

        # 随时保存 best_model.pth
        if val_acc > trainer.best_acc:
            trainer.best_acc = val_acc
            print(f"--> New best accuracy ({trainer.best_acc:.4f}), saving best_model.pth!")
            best_model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
            trainer.save_checkpoint(best_model_path, epoch + 1, model_config)

if __name__ == "__main__":
    main()
