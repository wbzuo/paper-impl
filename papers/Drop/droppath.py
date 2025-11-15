import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    随机丢弃该分支上的每个样本
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (batch_size, 1, 1, 1)维数与输入保持一致，仅需要batch_size个值
        mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        mask.floor_()  # 二值化，向下取整用于确定保存哪些样本

        output = x.div(keep_prob) * mask
        return output


if __name__ == "__main__":
    x = torch.randn((8, 197, 768))  # [batch_size, num_token, token_dim]
    drop_path = DropPath(drop_prob=0.5)
    my_o = drop_path(x)