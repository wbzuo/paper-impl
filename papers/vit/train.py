import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from datasets import load_transformed_dataset  # Assuming this loads the dataset with necessary transformations
from tqdm import tqdm
from model import VisionTransformer



def train_one_epoch(model, criterion ,optimizer ,train_loader, device, epoch):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # tqdm
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]")


    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate accuracy directly
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        correct = (predicted == labels).sum().item()  # Count correct predictions
        total_correct += correct  # Calculate accuracy for this batch
        total_samples += images.size(0)

        # Average loss and accuracy for the epoch
        # Update progress bar
        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        pbar.set_postfix({'Epoch': f'{epoch + 1}','Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.4f}'})

def validate(model, val_loader, criterion, device):
    
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"[Validation]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (outputs == labels).sum().item()
            total_correct += correct
            total_samples += imgs.size(0)

            # Collect preds and labels for confusion matrix
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.4f}'})

    avg_val_loss = val_loss / total_samples
    avg_val_accuracy = val_accuracy / total_samples
    return avg_loss, avg_acc, all_preds, all_labels



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs=20):
    """Full training pipeline"""
    # Record training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"Training started (Device: {device})")

    for epoch in range(epochs):
        # Train one epoch

        
        train_loss, train_acc = train_one_epoch(model, criterion ,optimizer ,train_loader, device, epoch)
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
    
    # Save trained model
    torch.save(model.state_dict(), 'vit_cifar10.pth')
    print(f"Model saved as 'vit_cifar10.pth'")
    
    return history, all_preds, all_labels


def plot_training_history(history, save_path='training_history.png'):
    """Plot training/validation loss and accuracy curves"""
    plt.figure(figsize=(12, 4))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.title('Training & Validation Loss', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2, color='blue')
    plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2, color='red', linestyle='--')
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy', fontsize=10)
    plt.title('Training & Validation Accuracy', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path='confusion_matrix.png'):
    """Plot confusion matrix for validation set"""
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (CIFAR-10 Validation Set)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def plot_predictions(model, val_loader, class_names, device, save_path='predictions.png'):
    """Plot sample predictions (correct and incorrect)"""
    model.eval()
    correct_samples = []
    incorrect_samples = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            
            # Collect samples
            for img, label, pred in zip(imgs, labels, preds):
                if len(correct_samples) < 5 and label == pred:
                    correct_samples.append((img.cpu(), label.cpu(), pred.cpu()))
                elif len(incorrect_samples) < 5 and label != pred:
                    incorrect_samples.append((img.cpu(), label.cpu(), pred.cpu()))
            
            if len(correct_samples) >= 5 and len(incorrect_samples) >= 5:
                break
    
    # Plot
    plt.figure(figsize=(15, 10))
    
    # Correct predictions
    for i, (img, label, pred) in enumerate(correct_samples):
        # Denormalize image
        img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])
        img = torch.clip(img, 0, 1)
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.title(f'Correct\nTrue: {class_names[label]}\nPred: {class_names[pred]}', fontsize=10)
        plt.axis('off')
    
    # Incorrect predictions
    for i, (img, label, pred) in enumerate(incorrect_samples):
        img = img.permute(1, 2, 0)
        img = img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])
        img = torch.clip(img, 0, 1)
        
        plt.subplot(2, 5, i+6)
        plt.imshow(img)
        plt.title(f'Incorrect\nTrue: {class_names[label]}\nPred: {class_names[pred]}', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('ViT Predictions on CIFAR-10 (Top: Correct, Bottom: Incorrect)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {save_path}")


def plot_patch_attention(model, val_loader, class_names, device, save_path='attention_visualization.png'):
    """Visualize attention weights between class token and patches"""
    model.eval()
    
    # Get one batch of data
    imgs, labels = next(iter(val_loader))
    imgs, labels = imgs.to(device), labels.to(device)
    
    # Forward pass up to patch embedding
    patch_embed = model.patch_embed(imgs)  # (B, N+1, D)
    
    # Get attention weights from first Transformer layer
    first_encoder_layer = model.encoder_layers[0]
    ln1_output = first_encoder_layer.ln1(patch_embed)
    q = first_encoder_layer.mhsa.q_proj(ln1_output)
    k = first_encoder_layer.mhsa.k_proj(ln1_output)
    
    B, N, D = q.shape
    num_heads = first_encoder_layer.mhsa.num_heads
    head_dim = D // num_heads
    
    # Reshape Q and K for attention calculation
    q = q.view(B, N, num_heads, head_dim).transpose(1, 2)  # (B, H, N, d)
    k = k.view(B, N, num_heads, head_dim).transpose(1, 2)  # (B, H, N, d)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
    attn_weights = torch.softmax(scores, dim=-1)  # (B, H, N, N)
    
    # Take class token (index 0) attention weights for first sample and first head
    sample_idx = 0
    head_idx = 0
    class_token_attn = attn_weights[sample_idx, head_idx, 0, 1:]  # (N,) -> exclude class token itself
    
    # Reshape attention weights to match patch grid
    patch_size = model.patch_embed.patch_size
    img_size = model.patch_embed.img_size
    grid_size = img_size // patch_size
    attn_grid = class_token_attn.view(grid_size, grid_size).cpu().numpy()
    
    # Get original image (denormalized)
    img = imgs[sample_idx].cpu().permute(1, 2, 0)
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]) + torch.tensor([0.4914, 0.4822, 0.4465])
    img = torch.clip(img, 0, 1).numpy()
    
    # Plot image + attention heatmap
    plt.figure(figsize=(10, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f'Original Image\nClass: {class_names[labels[sample_idx].cpu().item()]}', fontsize=12)
    plt.axis('off')
    
    # Attention heatmap
    plt.subplot(1, 2, 2)
    im = plt.imshow(attn_grid, cmap='hot', interpolation='bilinear')
    plt.colorbar(im, label='Attention Weight')
    plt.title('Class Token Attention Heatmap\n(Patch Importance)', fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Attention visualization saved to {save_path}")



from dataclasses import dataclass


# ======================== Main Function (One-Click Run) ========================
if __name__ == "__main__":
    # Hyperparameters
    config = {
         # model params
        'img_size': 224,
        'in_channels':3,
        'num_heads':8,
        'hidden_dim':1024,
        'dropout':0.1,
        'patch_size': 16,
        'embed_size': 768,
        'img_size': 224,
        'num_layers': 6,
        'n_classes': 10,

        # dataset params
        'batch_size': 64,

        # train process params
        'epochs': 10,
        'learning_rate': 1e-2,
        'weight_decay': 1e-5,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }
    # 
    from config import get_config
    config = get_config()
    # Step 1: Load data
    train_loader, test_loader = load_transformed_dataset(img_size = config.img_size ,batch_size=config.batch_size)
    
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
    print(f"ViT model initialized. Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
#     # Step 3: Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Classification task
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay
    )
    # Learning rate scheduler (reduce LR when val loss plateaus)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, 
        # verbose=True
    )
    
    # Step 4: Train model
    # a, b = train_one_epoch(model, criterion ,optimizer ,train_loader, config.device, config.epochs)
    # 

    # print(a, b)
    history, all_preds, all_labels = train_model(
        model, train_loader, test_loader, criterion, optimizer, scheduler, config.device, config.epochs
    )
    
    # Step 5: Load best model (optional, since we save the final model)
    model.load_state_dict(torch.load('vit_cifar10.pth', map_location=config.device))
    
    # Step 6: Generate diverse result visualizations
    plot_training_history(history)
    plot_confusion_matrix(all_labels, all_preds, class_names)
    plot_predictions(model, val_loader, class_names, DEVICE)
    plot_patch_attention(model, val_loader, class_names, DEVICE)
    
    # Step 7: Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    print("="*50)
    print("All processes completed successfully!")
    print(f"Generated files:")
    print(f"1. Trained model: vit_cifar10.pth")
    print(f"2. Training history: training_history.png")
    print(f"3. Confusion matrix: confusion_matrix.png")
    print(f"4. Sample predictions: predictions.png")
    print(f"5. Attention visualization: attention_visualization.png")
    print("="*50) 
print("Training complete.")
