import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768, img_size: int = 224):
        self.patch_size = patch_size
        super().__init__()
        # 做法1.使用卷积层(b, 14 * 14, 3 * 16 * 16)
        self.projection = nn.Sequential(
            # 使用一个卷积层而不是一个线性层 -> 性能增加
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # 将卷积操作后的patch铺平
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
        
        # 位置编码
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) ** 2 + 1, emb_size))
                
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

# 改进版本     
class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int = 768, num_heads: int = 8, dropout: float = 0):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        # 使用单个矩阵一次性计算出queries,keys,values
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x : torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # 将queries，keys和values划分为num_heads
        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)  # 划分到num_heads个头上
        
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        
        # 在最后一个维度上相加
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling

        att = self.att_drop(att)

        
        # 在第三个维度上相加
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)

        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.projection(out)
        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size: int, expansion: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )

class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))
       
        
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

        
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))

        
class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )


if __name__ == "__main__":
    print("start")
    input = torch.rand([128, 3, 32, 32])
    print(f"shape of input is {input.shape}")
    # model = PatchEmbedding(3, 16, 768)
    # output = model(input)
    # print(output.shape)
    # att = MultiHeadAttention()
    # output = att(output)
    model = ViT(in_channels = 3,
                patch_size = 4,
                emb_size = 384,
                img_size = 32,
                depth = 6,
                n_classes = 10)
    output = model(input)
    print(f"shape of output is {output.shape}")
    
    
    # 对数据进行patch化
    # patch_size = 16
    # patches = rearrange(input, 'b c (w s1) (h s2) -> b (h w) (s1 s2 c)', s1 = patch_size, s2 = patch_size)
    # print(f"shape of patches is {patches.shape}")
    print("end")