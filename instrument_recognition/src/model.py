import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class AdaptiveInstrumentClassifier(nn.Module):
    """
    提升型乐器分类器：升级至 ResNet34 以提取更深维度的声学特征，
    加设更猛烈的 Dropout 及全连接来抵抗过拟合。
    """
    def __init__(self, num_classes=11, in_channels=1):
        super(AdaptiveInstrumentClassifier, self).__init__()
        
        # 加载带预训练权重的 ResNet34
        self.backbone = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        # 我们的 Mel 频谱图是 1 通道的灰度图，为了不破坏预训练特征，我们在 dataset 中将其复制为 3 通道
        # 主干网络不用修改 conv1，完美复用 ImageNet 的底层特征
            
        # 修改全连接输出层，匹配 11 个乐器类别。这里使用带单一定量 Dropout 的简单线性分类器
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3), # 降低 Dropout
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, in_channels, n_mels, time_steps)
        # 比如：(32, 1, 128, 43) 
        # 我们默认将时间步和梅尔频率看作高度和宽度
        return self.backbone(x)
