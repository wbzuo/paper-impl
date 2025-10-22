import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from datasets import load_transformed_dataset  # Assuming this loads the dataset with necessary transformations
from tqdm import tqdm

from model import ViT  # For showing training progress

# Hyperparameters
config = {
    'patch_size': 4,
    'emb_size': 512,
    'img_size': 32,
    'depth': 6,
    'n_classes': 10,
    'batch_size': 512,
    'epochs': 10,
    'learning_rate': 1e-3,
}

# Initialize model, dataset, and dataloaders
model = ViT(
    in_channels=3,
    patch_size=config['patch_size'],
    emb_size=config['emb_size'],
    img_size=config['img_size'],
    depth=config['depth'],
    n_classes=config['n_classes']
)

# Load datasets
train_loader, test_loader = load_transformed_dataset(batch_size=config['batch_size'])

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(config['epochs']):
    model.train()
    running_loss = 0.0
    running_accuracy = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
        images, labels = images.to(device), labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate accuracy directly
        _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
        correct = (predicted == labels).sum().item()  # Count correct predictions
        accuracy = correct / labels.size(0)  # Calculate accuracy for this batch
        running_accuracy += accuracy

    # Average loss and accuracy for the epoch
    avg_loss = running_loss / len(train_loader)
    avg_accuracy = running_accuracy / len(train_loader)

    print(f"Epoch [{epoch+1}/{config['epochs']}] - Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")

    # Validation step (optional)
    model.eval()
    val_loss = 0.0
    val_accuracy = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0)
            val_accuracy += accuracy

    avg_val_loss = val_loss / len(test_loader)
    avg_val_accuracy = val_accuracy / len(test_loader)
    print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {avg_val_accuracy:.4f}")

print("Training complete.")
