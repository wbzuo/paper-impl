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

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    
    def forward(self, x):
        identity = x
        
        
        # main branch
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        # 1x1 conv shortcut branch
        if self.downsample is not None:
            identity = self.downsample(x)

        # fusion
        out += identity
        out = self.relu(out)
        
        return out


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
            



# With square kernels and equal stride
class ResNet(nn.Module):
 
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        # b c(3) h w
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        
                
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # fc层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
 
        # 网络刚开始训练时梯度更稳定，收敛更快。
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # 根据论文实验，可以提升模型 0.2~0.3% 的准确率。
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
 
        return x



class PatchUpModel(nn.Module):
    def __init__(self, model, num_classes = 10, block_size=7, gamma=.9, patchup_type='hard',keep_prob=.9):
        super().__init__()
        self.patchup_type = patchup_type
        self.block_size = block_size
        self.gamma = gamma
        self.gamma_adj = None
        self.kernel_size = (block_size, block_size)
        self.stride = (1, 1)
        self.padding = (block_size // 2, block_size // 2)
        self.computed_lam = None
        
        self.model = model
        self.num_classes = num_classes
        self. module_list = []
        for n,m in self.model.named_modules():
            if n[:-1]=='layer':
            #if 'conv' in n:
                self.module_list.append(m)

    def adjust_gamma(self, x):
        return self.gamma * x.shape[-1] ** 2 / \
               (self.block_size ** 2 * (x.shape[-1] - self.block_size + 1) ** 2)

    def forward(self, x, target=None):
        if target==None:
            out = self.model(x)
            return out
        else:

            self.lam = np.random.beta(2.0, 2.0)
            k = np.random.randint(-1, len(self.module_list))
            self.indices = torch.randperm(target.size(0)).cuda()
            self.target_onehot = to_one_hot(target, self.num_classes)
            self.target_shuffled_onehot = self.target_onehot[self.indices]
            
            if k == -1:  #CutMix
                W,H = x.size(2),x.size(3)
                cut_rat = np.sqrt(1. - self.lam)
                cut_w = np.int(W * cut_rat)
                cut_h = np.int(H * cut_rat)
                cx = np.random.randint(W)
                cy = np.random.randint(H)
        
                bbx1 = np.clip(cx - cut_w // 2, 0, W)
                bby1 = np.clip(cy - cut_h // 2, 0, H)
                bbx2 = np.clip(cx + cut_w // 2, 0, W)
                bby2 = np.clip(cy + cut_h // 2, 0, H)
                
                x[:, :, bbx1:bbx2, bby1:bby2] = x[self.indices, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
                out = self.model(x)
                loss = bce_loss(softmax(out), self.target_onehot) * lam +\
                    bce_loss(softmax(out), self.target_shuffled_onehot) * (1. - lam)
                
            else:
                modifier_hook = self.module_list[k].register_forward_hook(self.hook_modify)
                out = self.model(x)
                modifier_hook.remove()
                
                loss = 1.0 * bce_loss(softmax(out), self.target_a) * self.total_unchanged_portion + \
                     bce_loss(softmax(out), self.target_b) * (1. - self.total_unchanged_portion) + \
                    1.0 * bce_loss(softmax(out), self.target_reweighted)
            return out, loss
        
    def hook_modify(self, module, input, output):
        self.gamma_adj = self.adjust_gamma(output)
        p = torch.ones_like(output[0]) * self.gamma_adj
        m_i_j = torch.bernoulli(p)
        mask_shape = len(m_i_j.shape)
        m_i_j = m_i_j.expand(output.size(0), m_i_j.size(0), m_i_j.size(1), m_i_j.size(2))
        holes = F.max_pool2d(m_i_j, self.kernel_size, self.stride, self.padding)
        mask = 1 - holes
        unchanged = mask * output
        if mask_shape == 1:
            total_feats = output.size(1)
        else:
            total_feats = output.size(1) * (output.size(2) ** 2)
        total_changed_pixels = holes[0].sum()
        total_changed_portion = total_changed_pixels / total_feats
        self.total_unchanged_portion = (total_feats - total_changed_pixels) / total_feats
        if self.patchup_type == 'hard':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot +\
                total_changed_portion * self.target_shuffled_onehot
            patches = holes * output[self.indices]
            self.target_b = self.target_onehot[self.indices]
        elif self.patchup_type == 'soft':
            self.target_reweighted = self.total_unchanged_portion * self.target_onehot +\
                self.lam * total_changed_portion * self.target_onehot +\
                (1 - self.lam) * total_changed_portion * self.target_shuffled_onehot
            patches = holes * output
            patches = patches * self.lam + patches[self.indices] * (1 - self.lam)
            self.target_b = self.lam * self.target_onehot + (1 - self.lam) * self.target_shuffled_onehot
        else:
            raise ValueError("patchup_type must be \'hard\' or \'soft\'.")
        
        output = unchanged + patches
        self.target_a = self.target_onehot
        return output

        
# block测试    
# 假设输入为一个 batch 大小为 4 的特征图，通道为 64，尺寸为 56x56（与ResNet的常规配置一致）
x = torch.randn(4, 64, 56, 56)

# Case 1: 不改变通道数，不下采样
block1 = BasicBlock(in_channels=64, out_channels=64)
y1 = block1(x)
print("Case 1 output shape:", y1.shape)
# 预期输出: torch.Size([4, 64, 56, 56])

# Case 2: 通道数变化，需要downsample（使用1x1卷积匹配尺寸）
downsample = nn.Sequential(
    conv1x1(64, 128, stride=2),  # 调整通道 & 尺寸
    nn.BatchNorm2d(128)
)
block2 = BasicBlock(in_channels=64, out_channels=128, stride=2, downsample=downsample)
y2 = block2(x)
print("Case 2 output shape:", y2.shape)
# 预期输出: torch.Size([4, 128, 28, 28])




# ====================BottleNeck=========================
x = torch.randn(4, 64, 56, 56)

# Case 3: 不改变通道数，使用Bottleneck模块
# block1 = Bottleneck(in_channels=64, out_channels=64)
# y1 = block1(x)
# print("Case 3 output shape:", y1.shape)  # 预期输出: torch.Size([4, 256, 56, 56])

# # Case 4: 通道数变化，需要下采样
# downsample = nn.Sequential(
#     conv1x1(64, 256, stride=2),  # 1x1卷积，调整通道数并下采样空间尺寸
#     nn.BatchNorm2d(256)  # 对应的BatchNorm
# )
# block2 = Bottleneck(in_channels=64, out_channels=64, stride=2, downsample=downsample)
# y2 = block2(x)
# print("Case 4 output shape:", y2.shape)  # 预期输出: torch.Size([4, 256, 28, 28])


# # model = ResNet(Bottleneck, [3, 4, 6, 3], 1024)

# # input = torch.randn(128, 3, 32, 32)
# # print(f"shape of input is {input.shape}")
# # output = model(input)
# # print(f"shape of output is {output.shape}")



def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 创建ResNet-101模型
    # ResNet-18	BasicBlock	[2, 2, 2, 2]	
    # ResNet-34	BasicBlock	[3, 4, 6, 3]	
    # ResNet-50	Bottleneck	[3, 4, 6, 3]	
    # ResNet-101	Bottleneck	[3, 4, 23, 3]
    # ResNet-152	Bottleneck	[3, 8, 36, 3]	
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    model = PatchUpModel(model,num_classes=10, block_size=7, gamma=.9, patchup_type='hard')
    
    # 2. 将模型移动到设备
    model.to(device)
    
    # 3. 打印模型信息
    print("模型结构:")
    print(model)
    print(f"总参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 4. 创建随机输入数据
    batch_size = 128
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"\n输入张量形状: {x.shape}")
    
    # 5. 前向传播
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        y = model(x)
    
    # 6. 输出结果
    print(f"输出张量形状: {y.shape}")
    print(f"批次大小: {y.shape[0]}")
    print(f"类别数量: {y.shape[1]}")
    
    # 7. 验证输出范围（可选）
    print(f"输出值范围: [{y.min().item():.4f}, {y.max().item():.4f}]")

if __name__ == "__main__":
    main()