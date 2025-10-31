import torch
import torch.nn as nn

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias = False)
 

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias = False)



class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        # 第1个1x1卷积层 (用于改变通道数)
        self.conv1 = conv1x1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 第2个3x3卷积层 (用于提取空间特征)
        self.conv2 = conv3x3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 第3个1x1卷积层 (用于恢复通道数)
        self.conv3 = conv1x1(out_channels, out_channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace = True)
        
        
        self.stride = stride
        # 如果需要改变输入维度，则会用 downsample 层
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        # main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
        
        
        # shortcut branch
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        return out
            
            
            

x = torch.randn(4, 64, 56, 56)

# Case 3: 不改变通道数，使用Bottleneck模块
block1 = Bottleneck(in_channels=64, out_channels=64)
y1 = block1(x)
print("Case 3 output shape:", y1.shape)  # 预期输出: torch.Size([4, 256, 56, 56])

# Case 4: 通道数变化，需要下采样
downsample = nn.Sequential(
    conv1x1(64, 256, stride=2),  # 1x1卷积，调整通道数并下采样空间尺寸
    nn.BatchNorm2d(256)  # 对应的BatchNorm
)
block2 = Bottleneck(in_channels=64, out_channels=64, stride=2, downsample=downsample)
y2 = block2(x)
print("Case 4 output shape:", y2.shape)  # 预期输出: torch.Size([4, 256, 28, 28])