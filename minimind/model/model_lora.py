import torch
from torch import optim, nn

class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank  # LoRA的秩（rank），控制低秩矩阵的大小
        self.A = nn.Linear(in_features, rank, bias=False)  # 低秩矩阵A
        self.B = nn.Linear(rank, out_features, bias=False)  # 低秩矩阵B
        # 矩阵A高斯初始化
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        # 矩阵B全0初始化
        self.B.weight.data.zero_()
    
    def forward(self, x):
        return self.B(self.A(x))  # 顺序通过A和B矩阵
    

def apply_lora(model, rank=8):
    # 遍历模型的所有模块
    for name, module in model.named_modules():
        # 只对线性层且是方阵的权重应用LoRA
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA适配器
            lora = LoRA(module.weight.shape[0], module.weight.shape[1], rank=rank).to(model.device)
            # 将LoRA附加到原始模块
            setattr(module, "lora", lora)
            # 保存原始前向传播方法
            original_forward = module.forward

            # 定义新的前向传播（包含LoRA）
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)  # 原始输出 + LoRA适配输出

            # 替换前向传播方法
            module.forward = forward_with_lora
    

def load_lora(model, path):
    # 加载保存的LoRA权重
    state_dict = torch.load(path, map_location=model.device)
    # 遍历模型模块
    for name, module in model.named_modules():
        # 检查是否有LoRA适配器
        if hasattr(module, 'lora'):
            # 过滤出当前模块的LoRA权重
            lora_state = {k.replace(f'{name}.lora.', ''): v for k, v in state_dict.items() if f'{name}.lora.' in k}
            # 加载权重到LoRA适配器
            module.lora.load_state_dict(lora_state)

def save_lora(model, path):
    state_dict = {}
    # 遍历模型模块
    for name, module in model.named_modules():
        # 检查是否有LoRA适配器
        if hasattr(module, 'lora'):
            # 提取LoRA权重并添加前缀
            lora_state = {f'{name}.lora.{k}': v for k, v in module.lora.state_dict().items()}
            # 合并到总字典
            state_dict.update(lora_state)
    # 保存到文件
    torch.save(state_dict, path)
