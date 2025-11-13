import argparse
from dataclasses import dataclass, field
from typing import Optional, Tuple
import torch

@dataclass
class ViTConfig:
    """Vision Transformer configuration class."""
    
    # 图像参数
    img_size: int = 32
    patch_size: int = 4
    in_channels: int = 3
    
    # Transformer架构参数
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    dropout: float = 0.2
    attn_dropout: float = 0.0
    drop_path_rate: float = 0.0
    num_classes: int = 10

    
    # 训练参数
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 300
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    
    # 数据参数
    data_path: str = "./data"
    dataset: str = "imagenet"
    
    # 设备与随机种子
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42
    
    # 日志与保存
    log_dir: str = "./logs"
    save_dir: str = "./checkpoints"
    log_freq: int = 100
    
    def __post_init__(self):
        """验证配置参数。"""
        assert self.img_size % self.patch_size == 0, \
            f"Image size {self.img_size} must be divisible by patch size {self.patch_size}"
        assert self.embed_dim % self.num_heads == 0, \
            f"Embed dim {self.embed_dim} must be divisible by num_heads {self.num_heads}"

def get_config() -> ViTConfig:
    """从命令行参数获取配置。"""
    parser = argparse.ArgumentParser(description='Vision Transformer Training')
    
    # 模型架构参数组
    model_group = parser.add_argument_group('Model Architecture')
    model_group.add_argument('--img_size', type=int, default=224, 
                           help='Input image size')
    model_group.add_argument('--patch_size', type=int, default=16,
                           help='Patch size')
    model_group.add_argument('--embed_dim', type=int, default=768,
                           help='Embedding dimension')
    model_group.add_argument('--hidden_dim', type=int, default=1024,
                           help='MLP hidden dimension')                           
    model_group.add_argument('--num_layers', type=int, default=12,
                           help='Number of transformer layers')
    model_group.add_argument('--num_heads', type=int, default=12,
                           help='Number of attention heads')
    model_group.add_argument('--mlp_ratio', type=float, default=4.0,
                           help='MLP hidden dimension ratio')
    model_group.add_argument('--num_classes', type=int, default=1000,
                           help='Number of classes')
    
    # 训练参数组
    training_group = parser.add_argument_group('Training')
    training_group.add_argument('--batch_size', type=int, default=32,
                              help='Batch size')
    training_group.add_argument('--lr', type=float, default=1e-3,
                              help='Learning rate')
    training_group.add_argument('--epochs', type=int, default=300,
                              help='Number of epochs')
    training_group.add_argument('--weight_decay', type=float, default=1e-5,
                              help='Weight decay')
    training_group.add_argument('--warmup_epochs', type=int, default=10,
                              help='Warmup epochs')
    
    # 数据参数组
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data_path', type=str, default='./data',
                          help='Path to dataset')
    data_group.add_argument('--dataset', type=str, default='imagenet',
                          choices=['imagenet', 'cifar10', 'cifar100'],
                          help='Dataset name')
    
    # 其他参数
    other_group = parser.add_argument_group('Other')
    other_group.add_argument('--device', type=str, 
                           default='cuda' if torch.cuda.is_available() else 'cpu',
                           help='Device to use')
    other_group.add_argument('--seed', type=int, default=42,
                           help='Random seed')
    other_group.add_argument('--log_dir', type=str, default='./logs',
                           help='Log directory')
    other_group.add_argument('--save_dir', type=str, default='./checkpoints',
                           help='Checkpoint save directory')
    
    args = parser.parse_args()
    
    return ViTConfig(
        # 模型参数
        img_size=args.img_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        num_classes=args.num_classes,
        
        # 训练参数
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        
        # 其他参数
        device=args.device,
        seed=args.seed,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )