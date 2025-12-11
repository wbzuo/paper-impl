from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from typing import Optional, List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

# config parameters
class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"
    
    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = 'silu',
        hidden_size: int = 512,
        intermediate_size: Optional[int] = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: float = 1000000.0,
        inference_rope_scaling: bool = False,
        flash_attn: bool = True,
        # MOE specific configurations
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        
        # 基础模型参数
        self.dropout = dropout
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attn = flash_attn
        
        # MoE 参数
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        

# 这里使用Swish GELU

class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        
        # 自动计算中间层维度（如果未指定）
        if config.intermediate_size is None:
            # 计算基础中间层大小：隐藏层维度 × 8/3 ≈ 2.67倍扩展
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 对齐到64的倍数，优化硬件性能（内存对齐） + 向上取整
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            
                
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 门控投影
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # 上扩投影

        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 降维投影  
        
        
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


# Mixed of experts architecture
class MOEFeedForward(nn.Module):
    
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 创建路由专家
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config) # 门控
        
        # 创建共享专家（所有token都会使用）
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x  # 保存原始输入
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        
        
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            
            # 逐个专家处理
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            
            
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理时：使用优化的moe_infer方法
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        
        # 添加共享专家输出 
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss # 保存辅助损失
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x) # 初始化输出缓存
        # 1. 按专家索引排序，便于批量处理
        idxs = flat_expert_indices.argsort()
         # 2. 统计每个专家处理的token数量
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 3. 计算原始token索引（考虑每个token被多个专家处理）
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache



class MoEGate(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config

        self.top_k = config.num_experts_per_tok      # 每个token选择的专家数
        self.n_routed_experts = config.n_routed_experts  # 专家总数
        
        # 路由配置
        self.scoring_func = config.scoring_func      # 评分函数（softmax）

        self.alpha = config.aux_loss_alpha           # 辅助损失权重
        self.seq_aux = config.seq_aux                # 序列级辅助损失
        
        self.norm_topk_prob = config.norm_topk_prob  # 是否归一化top-k概率
        self.gating_dim = config.hidden_size         # 门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states: [bsz, sqe_len, h]
        bsz, seq_len, h = hidden_states.shape
        
        # 处理成单个 token [bsz * seq_len, h]
        hidden_states = hidden_states.view(-1, h)
        
        # 计算每个token与其他专家的匹配分数
        # logits形状: [batch_size * sequence_length, num_experts]
        # F.linear <==> hidden_states @ self.weight.T
        # [bsz * seq_len, h] @ [h, gate_size == [baz * seq_len, rexperts]
        logits = F.linear(hidden_states, self.weight, None)
        
        # 使用softmax将分数转换为概率分布
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)  # 维度: [batch_size * seq_len, num_experts]
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # 选择每个token最相关的top_k个专家
        # topk_weight: 选中的专家权重 [batch_size * seq_len, top_k]
        # topk_idx: 选中的专家索引 [batch_size * seq_len, top_k]
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果选择多个专家且启用概率归一化，确保选中的专家权重和为1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim = -1, keepdim = True) + 1e-20# 防止除零
            topk_weight = topk_weight / denominator  # 归一化权重

        # 计算负载均衡辅助损失（仅在训练时且alpha>0时计算）
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores  # 用于辅助损失的分数
            aux_topk = self.top_k    # 辅助损失考虑的top_k数
            
            # 将专家索引重塑为: [batch_size, sequence_length * top_k]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            
            if self.seq_aux:
                # 序列级辅助损失 - 确保每个batch内专家使用均衡
                # 将分数重塑为: [batch_size, sequence_length, num_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                
                # 初始化专家选择计数矩阵: [batch_size, num_experts]
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                
                # 统计每个batch中每个专家被选中的次数
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device))
                
                # 归一化：实际选择频率 vs 均匀分布期望
                # 期望频率 = (seq_len * top_k) / num_experts
                ce.div_(seq_len * aux_topk / self.n_routed_experts)
                
                # 计算损失：专家选择频率差异 × 平均专家分数
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
                
            else:
                # Token级辅助损失 - 全局专家使用均衡
                # 创建one-hot编码的专家选择掩码
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                
                # 计算每个专家的全局选择概率: [num_experts]
                ce = mask_ce.float().mean(0)
                
                # 计算每个专家的平均分数: [num_experts]
                Pi = scores_for_aux.mean(0)
                
                # 计算负载均衡因子
                fi = ce * self.n_routed_experts
                
                # 计算辅助损失: 平均分数 × 负载均衡因子
                aux_loss = (Pi * fi).sum() * self.alpha
                
        else:
            # 推理阶段或不需要辅助损失时，返回0
            aux_loss = 0
            
        # 返回结果:
        # topk_idx: 每个token选择的专家索引 [batch_size * seq_len, top_k]
        # topk_weight: 每个专家的权重 [batch_size * seq_len, top_k]  
        # aux_loss: 负载均衡损失标量
        return topk_idx, topk_weight, aux_loss


if __name__ == "__main__":
    # test
    config = MiniMindConfig()
    x = torch.randn(128, 4, 512)
    print("test FFN")
    model = FeedForward(config)
    y = model(x)
    print(f"shape of input:{x.shape} and shape of output: {y.shape}")

    model = MOEFeedForward(config)
    print("test MOEFFN")
    y = model(x)
    print(f"shape of input:{x.shape} and shape of output: {y.shape}")
