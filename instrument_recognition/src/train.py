# train.py
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from src.config import config
from src.data import get_dataloaders
from src.model import SimplifiedAdvancedClassifier

def train():
    device = config.DEVICE
    train_loader, val_loader, classes = get_dataloaders(val_split=0.2)
    
    model = SimplifiedAdvancedClassifier(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    
    best_acc = 0.0
    checkpoint_path = os.path.join(config.MODEL_DIR, "best_model.pth")
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
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
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({'loss': train_loss / (pbar.n + 1), 'acc': 100. * correct / total})

        # Eval
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        acc = 100. * correct / total
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best model to {checkpoint_path}")

if __name__ == "__main__":
    train()