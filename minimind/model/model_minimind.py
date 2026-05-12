import torch
import torch.nn as nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()  # 修正：应该是__init__而不是_init_
        self.eps = eps
        # 初始化可学习的缩放参数，初始值为全1，维度为dim
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x):
        # RMSNorm的核心计算：
        # mean(-1, keepdim=True) keepdim 表示 会在均值的维度填充1[2] -》[2, 1]
        # torch.rsqrt - 计算平方根的倒数，即1/sqrt(值)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x):
        # 1. 将输入转换为float类型进行归一化计算，确保数值稳定性
        # 2. 调用_norm方法进行RMS归一化
        # 3. 将结果转换回输入x的数据类型
        # 4. 乘以可学习的权重参数，恢复模型的表达能力
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


from transformers import PretrainedConfig


class MiniMindConfig(PretrainedConfig):
    model_type = "minimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: int = 1000000.0,
            inference_rope_scaling: bool = False,
            flash_attn: bool = True,
            ####################################################
            # Here are the specific configurations of MOE
            # When use_moe is false, the following is invalid
            ####################################################
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
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
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
        # 外推长度 = factor * original_max_position_embeddings
        self.rope_scaling = {
            "beta_fast": 4,
            "beta_slow": 1,
            "factor": 4,
            "original_max_position_embeddings": 2048,
            "type": "yarn"
        } if self.inference_rope_scaling else None
        self.flash_attn = flash_attn
        ####################################################
        # Here are the specific configurations of MOE
        # When use_moe is false, the following is invalid
        ####################################################
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率



import math
import torch
import torch.nn.init as init
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import random
import numpy as np
import torch
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    
    # 设置PyTorch的CPU随机种子，影响CPU上的所有随机操作
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        # 设置当前CUDA设备的随机种子
        torch.cuda.manual_seed(seed)
        # 设置所有CUDA设备的随机种子（多GPU情况下）
        torch.cuda.manual_seed_all(seed)
    
    # 强制CuDNN使用确定性算法，确保卷积操作每次结果一致
    # 注意：这可能会降低一些性能，但能保证可重复性
    torch.backends.cudnn.deterministic = True
    
    # 关闭CuDNN的自动优化基准测试
    # 避免CuDNN根据输入大小自动选择不同算法导致的随机性
    torch.backends.cudnn.benchmark = False
    
    # 设置CUDA的BLAS工作空间配置，避免某些操作中的非确定性行为
    # ':4096:8' 指定了工作空间的大小和配置
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 调用函数设置随机种子为42
set_seed(42)

  

# todo：这里Rotary Position Embeddings
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6,
                         rope_scaling: Optional[dict] = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    if rope_scaling is not None:
        orig_max, factor, beta_fast, beta_slow = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 4),
            rope_scaling.get("beta_fast", 4.0), rope_scaling.get("beta_slow", 1.0)
        )
        if end / orig_max > 1.0:
            corr_dim = next((i for i in range(dim // 2) if 2 * math.pi / freqs[i] > orig_max), dim // 2)
            power = torch.arange(0, dim // 2, device=freqs.device).float() / max(dim // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            # λ = (β·α - β + 1)/(β·α) YaRN标准公式
            scale = torch.where(torch.arange(dim // 2, device=freqs.device) < corr_dim, (beta * factor - beta + 1) / (beta * factor), 1.0 / factor)
            freqs = freqs * scale

    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed


from einops import repeat


# ========== 实现GQA =========================
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复键值对头的函数，用于在分组查询注意力(GQA)中扩展KV头到Q头的数量
    
    Args:
        x: 输入张量，形状为 [bs, seq_len, n_kv_heads, head_dim]
        n_rep: 重复次数，通常 n_rep = n_q_heads // n_kv_heads
    
    Returns:
        重复后的张量，形状为 [bs, seq_len, n_kv_heads * n_rep, head_dim]
    """
    # # 获取输入张量的形状维度
    # bs, seq_len, n_kv_heads, head_dim = x.shape
    
    # # 如果只需要重复1次，直接返回原张量
    # if n_rep == 1:
    #     return x
    
    # # 核心重复操作：
    # # 1. 首先在n_kv_heads和head_dim之间插入一个新维度
    # #    变为 [bs, seq_len, n_kv_heads, 1, head_dim]
    # return (x[:, :, :, None, :]
    #         # 2. 使用expand方法将新维度扩展到n_rep大小（不分配新内存）
    #         #    形状变为 [bs, seq_len, n_kv_heads, n_rep, head_dim]
    #         .expand(bs, seq_len, n_kv_heads, n_rep, head_dim)  # 修正：应该是expand不是extend
    #         # 3. 重新塑形，将n_kv_heads和n_rep维度合并
    #         #    形状变为 [bs, seq_len, n_kv_heads * n_rep, head_dim]
    #         .reshape(bs, seq_len, n_rep * n_kv_heads, head_dim))
    
    return repeat(x, 'b s n h -> b s (n n_rep) h', n_rep = n_rep)

class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        #
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        
        self.n_local_heads = args.num_attention_heads # Query头总数
        self.n_local_kv_heads = self.num_key_value_heads # Key-Value头总数  
        
        # 分组
        self.n_rep = self.n_local_heads // self.n_local_kv_heads # 分组个数
        
        self.head_dim = args.hidden_size // args.num_attention_heads # 每个头的维度
        
        # 权重矩阵
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        
        # 输出权重矩阵
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        
        # 定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        # 保存dropout概率
        self.dropout = args.dropout
        
        
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        
        
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        # RoPE
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现 缓存之前的K、V，避免重复计算
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        # 统一大小
        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            attn_mask = (
                None
                if attention_mask is None
                else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            )

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        
        # 自动计算中间层维度（如果未指定）
        if config.intermediate_size is None:
            # 计算基础中间层大小：隐藏层维度 × 8/3 ≈ 2.67倍扩展
            intermediate_size = int(config.hidden_size * 8 / 3)
            # 对齐到64的倍数，优化硬件性能（内存对齐）
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
            
                
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 门控投影
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 降维投影  
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)    # 上扩投影
        
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # 首先，输入x通过第一层线性变换和SILU激活函数
        # 然后，结果乘以输入x通过第三层线性变换的结果
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

# 测试

# 创建MLP实例
# args = MiniMindConfig()
# mlp = FeedForward(args)
# # 随机生成数据
# x = torch.randn(1, 50, 512)
# # 运行MLP模型
# output = mlp(x)
# print(output.shape)


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
        # 获取输入张量的形状: [bsz, sqe_len, h]
        bsz, seq_len, h = hidden_states.shape
        
        # 处理成单个 token
        hidden_states = hidden_states.view(-1, h)
        
        # 计算每个token与其他专家的匹配分数
        # logits形状: [batch_size * sequence_length, num_experts]
        # F.linear <==> hidden_states @ self.weight.T
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
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20  # 防止除零
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


class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value


class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, rope_base=config.rope_theta,
                                                    rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss


class MiniMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniMindConfig

    def __init__(self, config: MiniMindConfig = None):
        self.config = config or MiniMindConfig()
        super().__init__(self.config)
        self.model = MiniMindModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.model.embed_tokens.weight = self.lm_head.weight
        self.OUT = CausalLMOutputWithPast()

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                logits_to_keep: Union[int, torch.Tensor] = 0,
                **args):
        h, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(h[:, slice_indices, :])
        self.OUT.__setitem__('last_hidden_state', h)
        self.OUT.__setitem__('logits', logits)
        self.OUT.__setitem__('aux_loss', aux_loss)
        self.OUT.__setitem__('past_key_values', past_kvs)
        return self.OUT
    

if __name__ == "__main__":
    """测试MiniMind模型的输入输出形状"""
    print("开始测试 MiniMind 模型输入输出形状")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建配置
    config = MiniMindConfig(
        vocab_size=10000,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=2,
        num_hidden_layers=4,
        max_position_embeddings=2048,
        use_moe=True,
        n_routed_experts=4,
        num_experts_per_tok=2
    )
    
    # 创建模型
    model = MiniMindForCausalLM(config).to(device)
    model.eval()
    
    # 测试数据
    batch_size = 2
    seq_len = 16
    
    # 测试1: 完整序列前向传播
    print("\n1. 完整序列前向传播:")
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len)).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
    
    print(f"输入: {input_ids.shape}")
    print(f"输出logits: {outputs.logits.shape}")
    print(f"past_key_values: {len(outputs.past_key_values)} 个层")
    
    # 测试2: 自回归生成（使用缓存）
    print("\n2. 自回归生成:")
    next_input = torch.randint(0, config.vocab_size, (batch_size, 1)).to(device)
    
    with torch.no_grad():
        next_outputs = model(
            input_ids=next_input,
            past_key_values=outputs.past_key_values,
            use_cache=True
        )
    
    print(f"新输入: {next_input.shape}")
    print(f"新输出logits: {next_outputs.logits.shape}")
    
    # 测试3: 只保留最后几个logits
    print("\n3. 只保留最后logits:")
    with torch.no_grad():
        slice_outputs = model(
            input_ids=input_ids,
            logits_to_keep=3,  # 只保留最后3个token的logits
            use_cache=False
        )
    
    print(f"输入: {input_ids.shape}")
    print(f"切片输出logits: {slice_outputs.logits.shape}")
