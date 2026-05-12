import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import os

class CIFAR10Pair(datasets.CIFAR10):
    """CIFAR-10 Dataset returning positive pairs for SimCLR."""
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            x_i = self.transform(img)
            x_j = self.transform(img)
        else:
            x_i, x_j = img, img

        if self.target_transform is not None:
            target = self.target_transform(target)

        return x_i, x_j, target

def compute_train_transform():
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])
    ])
    return train_transform

def compute_test_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914,0.4822,0.4465],[0.2023,0.1994,0.2010])
    ])

# ---------------------------
# 2. SimCLR 模型
# ---------------------------
class SimCLR(nn.Module):
    def __init__(self, feature_dim=128):
        super(SimCLR, self).__init__()
        # ResNet18 backbone
        self.backbone = models.resnet18(pretrained=False)
        
        self.backbone.fc = nn.Identity()  # 去掉分类层
        dim_mlp = self.backbone.fc.in_features if hasattr(self.backbone.fc,'in_features') else 512

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, x):
        h = self.backbone(x)
        z = self.projection_head(h)
        return F.normalize(z, dim=1)

# ---------------------------
# 3. 对比损失
# ---------------------------
def simclr_loss(out_left, out_right, temperature=0.5):
    """NT-Xent loss (vectorized)."""
    batch_size = out_left.shape[0]
    out = torch.cat([out_left, out_right], dim=0)  # [2N, D]

    # 计算相似度矩阵
    sim_matrix = F.cosine_similarity(out.unsqueeze(1), out.unsqueeze(0), dim=2)
    
    # 对角线掩码
    mask = torch.eye(2*batch_size, dtype=torch.bool).to(out.device)
    sim_matrix = sim_matrix / temperature
    sim_matrix_exp = torch.exp(sim_matrix)
    sim_matrix_exp = sim_matrix_exp.masked_fill(mask, 0)

    # 正样本相似度
    positives = torch.exp(F.cosine_similarity(out_left, out_right) / temperature)
    positives = torch.cat([positives, positives], dim=0)

    loss = -torch.log(positives / sim_matrix_exp.sum(dim=1))
    return loss.mean()

# ---------------------------
# 4. 训练函数
# ---------------------------
from tqdm import tqdm
def train_simclr(model, loader, optimizer, device, epochs=1, temperature=0.5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_i, x_j, _ in tqdm(loader):
            x_i, x_j = x_i.to(device), x_j.to(device)
            z_i, z_j = model(x_i), model(x_j)
            loss = simclr_loss(z_i, z_j, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(loader):.4f}")

# ---------------------------
# 5. kNN 测试
# ---------------------------
@torch.no_grad()
def knn_test(model, memory_loader, test_loader, device, k=200):
    model.eval()
    memory_features, memory_labels = [], []
    for x_i, x_j, y in memory_loader:
        x_i = x_i.to(device)
        feats = model(x_i)
        memory_features.append(feats)
        memory_labels.append(y)
    memory_features = torch.cat(memory_features, dim=0)
    memory_labels = torch.cat(memory_labels, dim=0)

    top1, total = 0, 0
    for x_i, x_j, y in test_loader:
        x_i = x_i.to(device)
        feats = model(x_i)
        dists = torch.mm(feats, memory_features.t())  # cosine similarity
        _, idx = dists.topk(k, dim=1)
        preds = memory_labels[idx]
        # top-1 vote
        pred_labels = []
        for row in preds:
            counts = torch.bincount(row)
            pred_labels.append(torch.argmax(counts))
        pred_labels = torch.tensor(pred_labels).to(device)
        top1 += (pred_labels == y.to(device)).sum().item()
        total += y.size(0)
    print(f"Top-1 Accuracy: {top1/total*100:.2f}%")

# ---------------------------
# 6. 主函数
# ---------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 128
    epochs = 10
    feature_dim = 128



    data_root = r"C:\Users\Administrator\Desktop\251\study\datasets"


    
    # 数据集
    train_dataset = CIFAR10Pair(root=data_root, train=True, transform=compute_train_transform(), download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    memory_dataset = CIFAR10Pair(root=data_root, train=True, transform=compute_test_transform(), download=True)
    memory_loader = DataLoader(memory_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = CIFAR10Pair(root=data_root, train=False, transform=compute_test_transform(), download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 模型与优化器
    model = SimCLR(feature_dim=feature_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # 训练
    train_simclr(model, train_loader, optimizer, device, epochs=epochs, temperature=0.5)

    # kNN 测试
    knn_test(model, memory_loader, test_loader, device, k=200)
