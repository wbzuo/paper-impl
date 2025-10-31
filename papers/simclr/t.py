import numpy as np
import torch
from torchvision import transforms 
from torch.utils.data import DataLoader
import torchvision
import numpy as np
import matplotlib.pyplot as plt





# 一种神经网络基编码器f（·），从增广数据示例中提取表示向量。
# 我们的框架允许在没有任何约束的情况下选择各种网络架构。
# 我们选择简单，并采用常用的ResNet


# 一种小型神经网络投影头g（·），将表示映射到应用对比度损失的空间。
# 我们使用具有一个隐藏层的MLP来获得zi=g（hi）=W（2）σ（W（1）hi），
# 其中σ是ReLU非线性。如第4节所示，我们发现在字上而不是在hi上定义对比损失是有益的。




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"test seed {torch.rand(3)}")  # 每次运行应输出相同的随机数


