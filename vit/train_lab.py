import swanlab
import torch
import torch.nn as nn
import time
from tqdm import tqdm

from model import ViT

def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, config, swanlab_run=None):
    """训练模型多个epoch"""
    
    device = config.device
    model.to(device)
    
    # 训练历史记录
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'learning_rates': [], 'epoch_times': []
    }
    
    best_acc = 0.0
    
    print(f"🚀 Starting training for {config.epochs} epochs on {device}")
    print(f"📊 Training batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, epoch, device, config
        )
        
        # 测试一个epoch
        test_loss, test_acc = test_epoch(
            model, test_loader, criterion, device, config
        )
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['learning_rates'].append(current_lr)
        history['epoch_times'].append(epoch_time)
        
        # 打印epoch结果
        print(f"✅ Epoch: {epoch:02d}/{config.epochs} | "
              f"Time: {epoch_time:.2f}s | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | "
              f"LR: {current_lr:.6f}")
        
        # SwanLab记录
        if swanlab_run is not None:
            swanlab.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "learning_rate": current_lr,
                "epoch_time": epoch_time
            })
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'test_acc': test_acc,
                'train_acc': train_acc,
                'config': config
            }, 'best_vit_model.pth')
            print(f"💾 New best model saved! Test Accuracy: {test_acc:.2f}%")
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best Test Accuracy: {best_acc:.2f}%")
    
    return history, best_acc

def train_epoch(model, train_loader, criterion, optimizer, epoch, device, config):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # 使用tqdm进度条
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} Training', leave=False)
    
    for batch_idx, (data, targets) in enumerate(pbar):
        # 将数据转移到指定设备
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（可选）
        if hasattr(config, 'grad_clip'):
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        # 统计信息
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        current_acc = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{current_acc:.2f}%'
        })
        
        # 定期打印
        if batch_idx % config.get('log_interval', 100) == 0:
            print(f'   Batch: {batch_idx:03d}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f} | Acc: {current_acc:.2f}%')
    
    pbar.close()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    
    return avg_loss, accuracy

def test_epoch(model, test_loader, criterion, device, config):
    """测试一个epoch"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Testing', leave=False)
        
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 更新进度条
            current_acc = 100. * correct / total
            pbar.set_postfix({
                'Acc': f'{current_acc:.2f}%'
            })
        
        pbar.close()
    
    accuracy = 100. * correct / total
    avg_loss = test_loss / len(test_loader)
    
    return avg_loss, accuracy

# 使用示例
def main():
    # 初始化配置
    config = {
        "learning_rate": 0.001,
        "architecture": "vit",
        "dataset": "CIFAR-10",
        "epochs": 10,
        "batch_size": 128,
        "img_size": 32,
        "patch_size": 4,
        "emb_size": 384,
        "depth": 6,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_interval": 100,
        "grad_clip": 1.0
    }
    
    # 创建模型
    model = ViT(
        in_channels=3,
        patch_size=config['patch_size'],
        emb_size=config['emb_size'],
        img_size=config['img_size'],
        depth=config['depth'],
        n_classes=10
    )
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs']
    )
    
    # 开始训练
    history, best_acc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        swanlab_run=run  # 传入SwanLab run对象
    )

if __name__ == "__main__":
    main()