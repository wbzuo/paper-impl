import torch
import torch.nn as nn
import torch.nn.functional as F

class LoraLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        test_mode: bool = False
    ):
        super().__init__()
        self.base_layer = base_layer  # 原始线性层
        self.r = r                    # LoRA 秩
        self.alpha = alpha            # 缩放系数
        self.scaling = alpha / r      # 缩放因子
        # 占位层不做任何操作（可用于Ablation Study）
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.test_mode = test_mode    # 测试模式（是否启用LoRA）
        
        # 定义A B矩阵
        self.lora_A = nn.Parameter(torch.empty(r, base_layer.in_features, dtype=base_layer.weight.dtype))
        self.lora_B = nn.Parameter(torch.empty(base_layer.out_features, r, dtype=base_layer.weight.dtype))
        
        # 初始化lora矩阵
        nn.init.normal_(self.lora_A, mean=0.0, std=0.02)
        if test_mode:
            nn.init.normal_(self.lora_B, mean=0.0, std=0.02)
        else:
            nn.init.zeros_(self.lora_B)

        # 冻结原来的层的参数
        for param in self.base_layer.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scaling = float(self.alpha) / float(self.r)     # lora 缩放系数
        lora_adjustment = F.linear(self.dropout(x), self.lora_A)
        lora_adjustment = F.linear(lora_adjustment, self.lora_B)
        return self.base_layer(x) + lora_adjustment * scaling
    


def replace_linear_with_lora(
    module: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout_p: float = 0.0,
    embed_requires_grad: bool = False,      # embedding 层是否训练
    norm_requires_grad: bool = False,       # norm 层是否训练
    head_requires_grad: bool = False,       # lm_head 层是否训练（Causal LM才有）
    test_mode: bool = False,                # 测试模式，用于控制 lora_B 是否为全零
):
    """
    找到 module 中所有线性层并递归替换
    """
    for name, child in module.named_children():
        # 先处理额外的层，lm_head 也是 linear，所以先处理
        if any(s in name for s in ['embed', 'norm', 'lm_head']):
            requires_grad = embed_requires_grad if 'embed' in name \
                            else norm_requires_grad if 'norm' in name \
                            else head_requires_grad
            for param in child.parameters():
                param.requires_grad = requires_grad
        # 替换所有线性层，QLoRA 做法
        elif isinstance(child, nn.Linear):
            lora_linear = LoraLinear(child, r=r, alpha=alpha, dropout_p=dropout_p, test_mode=test_mode)
            setattr(module, name, lora_linear)
        # 递归向下替换
        else:
            replace_linear_with_lora(
                child, r, alpha, dropout_p,
                embed_requires_grad, norm_requires_grad, head_requires_grad,
                test_mode=test_mode
            )

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