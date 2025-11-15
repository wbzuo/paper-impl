import torch.nn as nn
import torch


class dropout(nn.Module):
    def __init__(self, drop_prob):
        super(dropout, self).__init__()
        assert 0 <= drop_prob <= 1, 'drop_prob should be [0, 1]'
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training:
            keep_prob = 1 - self.drop_prob
            mask = keep_prob + torch.rand(x.shape)
            mask.floor_()
            return x.div(keep_prob) * mask
        else:
            return x


if __name__ == '__main__':
    x = torch.randn((8, 768))  # [batch_size, feat_dim]，dropout常在全连接层之后，所以我们以一维数据为例
    drop = dropout(0.1)
    my_o = drop(x)