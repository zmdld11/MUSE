import os
import torch
import numpy as np
from tqdm import tqdm
from config import Config

class ModelTrainer:
    def __init__(self, model, optimizer, criterion, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        
        self.best_acc = 0.0
        self.start_epoch = 0

    def resume_from_checkpoint(self, checkpoint_path):
        """Loads a checkpoint if available."""
        if os.path.exists(checkpoint_path):
            print(f"=> Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint.get('best_acc', 0.0)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            print(f"=> Loaded checkpoint (epoch {self.start_epoch}, best_acc {self.best_acc:.4f})")
            return True
        return False

    def save_checkpoint(self, path, epoch, model_config):
        """Saves current state to a checkpoint."""
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'best_acc': self.best_acc,
            'optimizer_state': self.optimizer.state_dict(),
            'model_name': 'AdaptiveInstrumentClassifier',
            'config': model_config
        }, path)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        
        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # 数据增强：MixUp 极其适合频谱图的高效抗过拟合策略
            if torch.rand(1).item() > 0.5:
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs.size(0)).to(self.device)
                inputs = lam * inputs + (1 - lam) * inputs[index]
                targets_a, targets_b = targets, targets[index]
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = lam * self.criterion(outputs, targets_a) + (1 - lam) * self.criterion(outputs, targets_b)
            else:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        return train_loss / len(train_loader)

    def val_epoch(self, val_loader, epoch):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]")
        with torch.no_grad():
            for inputs, targets in pbar_val:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        return avg_val_loss, val_acc