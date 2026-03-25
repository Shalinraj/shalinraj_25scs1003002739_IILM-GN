import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

print("🔍 Checking device...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

data_dir = "dataset"

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

print("📂 Loading dataset...")
train_dataset = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,          # IMPORTANT FOR WINDOWS
    pin_memory=True
)

print("Found classes:", train_dataset.classes)
print("Total images:", len(train_dataset))

print(f"🔍 Loader length: {len(train_loader)} batches")

# Load pre-trained ResNet
from torchvision.models import ResNet18_Weights
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Modify final layer
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 5
print("🔥 Training started...\n")

for epoch in range(num_epochs):
    running_loss = 0.0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Print loss every 20 batches
        if (batch_idx + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Step [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")

    print(f"\nEpoch [{epoch+1}/{num_epochs}] Completed ✔ | Avg Loss: {running_loss/len(train_loader):.4f}\n")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/leaf_model.pth")

print("🎉 Training completed successfully!")
print("📁 Saved model as: models/leaf_model.pth")
