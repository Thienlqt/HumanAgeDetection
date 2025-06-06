import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import matplotlib.pyplot as plt

# Configurations
TRAIN_DIR = "/kaggle/input/Age_Dataset/Training"
VAL_DIR = "/kaggle/input/Age_Dataset/Validation"
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "/kaggle/working/age_prediction_checkpoint.pth"
FINAL_MODEL_PATH = "/kaggle/working/age_prediction_resnet50.pth"

# Define age bins
AGE_BINS = ['0-10', '11-20', '21-30', '31-40', '41-50', '51-60', '61-70', '71-80']

# Data Transforms
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Dataset
class AgeDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for label, age_bin in enumerate(sorted(os.listdir(root_dir))):
            bin_dir = os.path.join(root_dir, age_bin)
            if os.path.isdir(bin_dir):
                for img_file in os.listdir(bin_dir):
                    img_path = os.path.join(bin_dir, img_file)
                    if os.path.isfile(img_path):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

# Dataloaders
def load_dataloader(folder_path, batch_size, transform):
    dataset = AgeDataset(folder_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_loader = load_dataloader(TRAIN_DIR, BATCH_SIZE, train_transform)
val_loader = load_dataloader(VAL_DIR, BATCH_SIZE, val_transform)

# Load Pretrained Model and Modify
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, len(AGE_BINS))
model = model.to(DEVICE)

# Loss, Optimizer, and Scheduler
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

# Load checkpoint if it exists
def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch + 1}")
        return start_epoch
    print("No checkpoint found. Starting training from scratch.")
    return 0

def train_and_finetune_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, checkpoint_path, final_model_path):
    start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    for epoch in range(start_epoch, epochs):
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_train_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        train_losses.append(train_loss)

        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_val_loss / len(val_loader)
        val_accuracy = correct_val / total_val
        val_losses.append(val_loss)

        scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} \n"
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} \n"
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), final_model_path)

    print(f"Model training completed. Best model saved to {final_model_path}")
    return train_losses, val_losses

# Train and Fine-tune
train_losses, val_losses = train_and_finetune_model(
    model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS, CHECKPOINT_PATH, FINAL_MODEL_PATH
)

# Plot the Loss Curves
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
