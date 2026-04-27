# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return self.dropout(out)

class AttentionModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(AttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SimplifiedAdvancedClassifier(nn.Module):
    def __init__(self, num_classes=11):
        super(SimplifiedAdvancedClassifier, self).__init__()
        
        self.in_channels = 32
        self.init_conv = nn.Sequential(
            nn.Conv2d(1, self.in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.attn1 = AttentionModule(64, 2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.attn2 = AttentionModule(128, 4)
        pass # To simplify code but retain main structure
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)
        
    def forward(self, x):
        out = self.init_conv(x)
        out = self.attn1(self.layer1(out))
        out = self.attn2(self.layer2(out))
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)
