import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class CIFAR10Pair(CIFAR10):
    """Return two random augmented views of the same image for SimCLR."""
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
        transforms.RandomApply(torch.nn.ModuleList([color_jitter]), p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.GaussianBlur(kernel_size=3),  # 模糊增强，SimCLR中常用
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])
    return train_transform


def compute_test_transform():
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465],
                             [0.2023, 0.1994, 0.2010])
    ])
    return test_transform



def get_dataloader(img_size=32, batch_size=128):
    train_transform = compute_train_transform()
    test_transform = compute_test_transform()

    data_root = r"C:\Users\Administrator\Desktop\251\study\datasets"

    train_dataset = CIFAR10Pair(root=data_root, train=True, download=False, transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader



def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: t * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
                                      + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)),
        transforms.Lambda(lambda t: torch.clamp(t, 0, 1)),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW → HWC
    ])

    if len(image.shape) == 4:
        image = image[0]  # 取batch中第一张图

    img = reverse_transforms(image).numpy()
    return img



if __name__ == "__main__":
    train_loader, test_loader = get_dataloader()

    # 从训练集中取出一批数据
    x_i, x_j, target = next(iter(train_loader))

    # 显示同一图像的两个随机增强视图
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(show_tensor_image(x_i))
    axes[0].set_title("Augmented view 1")
    axes[0].axis("off")

    axes[1].imshow(show_tensor_image(x_j))
    axes[1].set_title("Augmented view 2")
    axes[1].axis("off")

    plt.show()


    