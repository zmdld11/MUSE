import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class AdaptiveInstrumentClassifier(nn.Module):
    """
    通用型基类乐器分类器：只要通过此类名动态实例化，
    无论多少类，输入多少通道，或者特征维度，只要改变 config 即可，
    不需要动用推断代码。
    """
    def __init__(self, num_classes=11, in_channels=1):
        super(AdaptiveInstrumentClassifier, self).__init__()
        
        # 加载带预训练权重的 ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # 我们的 Mel 频谱图是 1 通道的灰度图（或者是伪彩色的 3 通道），修改第 1 个卷积层
        if in_channels != 3:
            self.backbone.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # 修改全连接输出层，匹配 11 个乐器类别
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5), # 防止过拟合
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, in_channels, n_mels, time_steps)
        # 比如：(32, 1, 128, 43) 
        # 我们默认将时间步和梅尔频率看作高度和宽度
        return self.backbone(x)
