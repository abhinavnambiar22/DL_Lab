import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import DataLoader
import os

# Step 1: Define Data Transformations (with Augmentation for Training)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Resize images
    transforms.RandomHorizontalFlip(),  # Flip horizontally with 50% probability
    transforms.RandomRotation(10),   # Rotate images randomly within ±10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Random color changes
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for consistency
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Step 2: Load Dataset
data_dir = "cats_and_dogs_filtered"
train_dir = os.path.join(data_dir, "train")
valid_dir = os.path.join(data_dir, "validation")

train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transform)
valid_dataset = torchvision.datasets.ImageFolder(root=valid_dir, transform=valid_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Step 3: Load Pretrained AlexNet Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
alexnet = models.alexnet(pretrained=True)

# Step 4: Modify the Classifier for Binary Classification
num_ftrs = alexnet.classifier[6].in_features
alexnet.classifier[6] = nn.Linear(num_ftrs, 2)  # 2 classes: Cat and Dog
alexnet = alexnet.to(device)

# Step 5: Define Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(alexnet.classifier[6].parameters(), lr=0.001)

# Step 6: Train the Model
num_epochs = 5
for epoch in range(num_epochs):
    alexnet.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = alexnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

# Step 7: Evaluate the Model
alexnet.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = alexnet(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct / total
print(f"Validation Accuracy: {accuracy:.2f}%")
