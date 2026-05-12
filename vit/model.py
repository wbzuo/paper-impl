import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3, 
        patch_size: int = 16, 
        emb_size: int = 768, 
        img_size: int = 224,
        ):
        super().__init__()
        self.patch_size = patch_size
        # 224 = 14 * 16
        # 做法1.使用卷积层(b, 14 * 14, 3 * 16 * 16) 主要做法
        
        self.projection = nn.Sequential(
            # 使用一个卷积层而不是一个线性层 -> 性能增加
            # b c h w -> b e  h // p  w // p
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            
            # 将卷积操作后的patch铺平
            # b e   h // p  w // p -> 
            Rearrange('b e h w -> b (h w) e'),
        )
        
        # 做法2.使用线性层实现
        # self.projection = nn.Sequential(
        #     # 将原始图像切分为16*16的patch并把它们拉平
        #     Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
        #     # 注意这里的隐层大小设置的也是768，可以配置
        #     nn.Linear(patch_size * patch_size * in_channels, emb_size)
        # )
        
        
        # 生成cls_token的emb_size 添加在序列最前面的
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        
        self.num_patches  = (img_size // patch_size) ** 2
        self.positions = nn.Parameter(torch.randn(1, self.num_patches+ 1, emb_size))

         
    def forward(self, x):
        b, _, _, _ = x.shape

        # patch操作
        x = self.projection(x)
        
        # cls_token拓展b次 (b, 1, emb_size)
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b = b)
        
        # cls_tokens和patch进行拼接
        x = torch.cat([cls_tokens, x], dim = 1)
        # 添加位置编码
        x += self.positions

        return x

input = torch.randn(128, 3, 224,224)
pathcEmbedding = PatchEmbedding()
output = pathcEmbedding(input)
print(f"shape of input is {input.shape}")
print(f"shape of output is {output.shape}")

# 注意力机制 # 官网实现
# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size = 512, num_heads = 8, dropout = 0):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads
#         self.keys = nn.Linear(emb_size, emb_size)
#         self.querys = nn.Linear(emb_size, emb_size)
#         self.values = nn.Linear(emb_size, emb_size)
#         self.att_drop = nn.Dropout(dropout)
#         self.projection = nn.Linear(emb_size, emb_size)
            
        
#     def forward(self, x, mask = None):
#         queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
#         keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
#         values  = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
#         energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
#         if mask is not None:
#             fill_value = torch.finfo(torch.float32).min
#             energy.mask_fill(~mask, fill_value)
            
#         scaling = self.emb_size ** (1/2)
#         att = F.softmax(energy, dim=-1) / scaling
#         att = self.att_drop(att)
#         # sum up over the third axis
#         out = torch.einsum('bhal, bhlv -> bhav ', att, values)
#         out = rearrange(out, "b h n d -> b n (h d)")
#         out = self.projection(out)
#         return out

# # 改进版本     
# class MultiHeadAttention(nn.Module):
#     def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
#         super().__init__()
#         self.emb_size = emb_size
#         self.num_heads = num_heads

#         # 确保emb_size可以被num_heads整除
#         assert self.head_dim * num_heads == emb_size, "emb_size must be divisible by num_heads"

#         # 使用单个矩阵一次性计算出queries,keys,values
#         self.qkv = nn.Linear(emb_size, emb_size * 3)
#         self.att_drop = nn.Dropout(dropout)
#         self.projection = nn.Linear(emb_size, emb_size)
        
#     def forward(self, x : torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
#         # 将queries，keys和values划分为num_heads
#         qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
        
#         queries, keys, values = qkv[0], qkv[1], qkv[2]
        
#         # 在最后一个维度上相加
#         energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
#         if mask is not None:
#             fill_value = torch.finfo(torch.float32).min
#             energy.mask_fill(~mask, fill_value)
        
#         scaling = self.emb_size ** (1/2)
#         att = F.softmax(energy / scaling, dim=-1) 

#         att = self.att_drop(att)

        
#         # 在第三个维度上相加
#         out = torch.einsum('bhal, bhlv -> bhav ', att, values)

#         out = rearrange(out, "b h n d -> b n (h d)")

#         out = self.projection(out)
#         return out

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) module
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, embed_dim=768, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head
        
        # Ensure embed_dim is divisible by num_heads
        assert self.head_dim * num_heads == embed_dim, "Embed dim must be divisible by num heads"
        
        # Linear layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output linear layer
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B, N, D = x.shape  # (B, 65, 256)
        
        # Step 1: Compute Q, K, V (B, N, D)
        q = self.q_proj(x)  # (B, 65, 256)
        k = self.k_proj(x)  # (B, 65, 256)
        v = self.v_proj(x)  # (B, 65, 256)
        
        # Step 2: Split into multiple heads (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 8, 65, 32)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 8, 65, 32)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, 8, 65, 32)
        
        # Step 3: Compute attention scores (B, num_heads, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1))  # (B, 8, 65, 65)
        scores = scores / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # Scale
        
        # Step 4: Softmax to get attention weights (B, num_heads, N, N)
        attn_weights = torch.softmax(scores, dim=-1)  # (B, 8, 65, 65)
        attn_weights = self.dropout(attn_weights)
        
        # Step 5: Compute weighted sum of V (B, num_heads, N, head_dim)
        attn_output = torch.matmul(attn_weights, v)  # (B, 8, 65, 32)
        
        # Step 6: Concatenate heads (B, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, 65, 8, 32)
        attn_output = attn_output.view(B, N, D)  # (B, 65, 256)
        
        # Step 7: Linear projection
        output = self.out_proj(attn_output)  # (B, 65, 256)
        output = self.dropout(output)
        
        return output

# input = torch.randn(128, 197, 768)
# mha = MultiHeadAttention()
# output = mha(input)
# print(f"shape of input is {input.shape}")
# print(f"shape of output is {output.shape}")



class MLP(nn.Module):
    """
    Multi-Layer Perceptron for Transformer encoder
    Input: (B, N, D)
    Output: (B, N, D)
    """
    expansion = 4

    def __init__(self, embed_dim=256, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)  # Expand to 4x dimension
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # Project back
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()  # Activation function
        
    def forward(self, x):
        x = self.fc1(x)  # (B, 65, 256) -> (B, 65, 1024)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 65, 1024) -> (B, 65, 256)
        x = self.dropout(x)
        return x
# input = torch.randn(128, 197, 768)

# mlp = MLP(768, 1024, 0.1)
# output = mlp(input)
# print(f"shape of input is {input.shape}")
# print(f"shape of output is {output.shape}")

class TransformerEncoderLayer(nn.Module):
    """
    Single layer of Transformer encoder
    Input: (B, N, D)
    Output: (B, N, D)
    """
    def __init__(self, embed_dim=256, num_heads=8, hidden_dim=1024, dropout=0.1):
        super().__init__()
        # Layer normalization before MHSA
        self.ln1 = nn.LayerNorm(embed_dim)
        # Multi-Head Self-Attention
        self.mha = MultiHeadAttention(embed_dim, num_heads, dropout)
        # Layer normalization before MLP
        self.ln2 = nn.LayerNorm(embed_dim)
        # MLP block
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        
    def forward(self, x):
        # Residual connection + MHSA
        x = x + self.mha(self.ln1(x))  # (B, 65, 256)
        # Residual connection + MLP
        x = x + self.mlp(self.ln2(x))  # (B, 65, 256)
        return x

# input = torch.randn(128, 197, 768)

# encoderLayer = TransformerEncoderLayer(768, 8, 1024, 0.1)
# output = encoderLayer(input)
# print(f"shape of input is {input.shape}")
# print(f"shape of output is {output.shape}")

class VisionTransformer(nn.Module):
    """
    Full Vision Transformer model for image classification
    Input: (B, C, H, W)
    Output: (B, num_classes)
    """
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_ch=3,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        hidden_dim=1024,
        num_classes=10,
        dropout=0.1
    ):
        super().__init__()
        # Patch embedding + class token + positional embedding
        self.patch_embed = PatchEmbedding(
            img_size = img_size, 
            patch_size = patch_size, 
            in_channels = in_ch, 
            emb_size = embed_dim
        )
        
        # Transformer encoder (stack multiple layers)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization for encoder output
        self.ln = nn.LayerNorm(embed_dim)
        
        # Classification head (linear layer)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """Initialize model weights (improve training stability)"""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
    
    def forward(self, x):
        # Step 1: Patch embedding + positional encoding (B, C, H, W) -> (B, N+1, D)
        x = self.patch_embed(x)  # (B, 65, 256)
        
        # Step 2: Pass through Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)  # (B, 65, 256)
        
        # Step 3: Layer normalization
        x = self.ln(x)  # (B, 65, 256)
        
        # Step 4: Extract class token feature (B, D)
        class_token_feature = x[:, 0, :]  # (B, 256)
        
        # Step 5: Classification head (B, num_classes)
        logits = self.classifier(class_token_feature)  # (B, 10)
        
        return logits

from config.base import get_config

if __name__ == "__main__":
    print("start")
    input = torch.rand([128, 3, 224, 224])
    print(f"shape of input is {input.shape}")
    # model = PatchEmbedding(3, 16, 768)
    # output = model(input)
    # print(output.shape)
    # att = MultiHeadAttention()
    # output = att(output)
    # Step 2: Initialize ViT model

    config = get_config()

    # Step 2: Initialize ViT model
    model = VisionTransformer(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_ch=config.in_channels,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        hidden_dim=int(config.embed_dim * config.mlp_ratio),
        num_classes=config.num_classes,
        dropout=config.dropout
    )
    device = config.device
    model.to(device)

    
    output = model(input)
    print(f"shape of output is {output.shape}")
    
    
    # 对数据进行patch化
    # patch_size = 16
    # patches = rearrange(input, 'b c (w s1) (h s2) -> b (h w) (s1 s2 c)', s1 = patch_size, s2 = patch_size)
    # print(f"shape of patches is {patches.shape}")
    print("end")