import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BrainMRIDataset
from model import UNet
from train import train_model
from evaluate import evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
train_transforms = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop((256, 256), scale=(0.95, 1.05)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Load dataset and create DataLoader
train_dataset = BrainMRIDataset(dataframe=df, image_transform=train_transforms, mask_transform=train_transforms)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model, optimizer, scheduler
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# Train the model
train_model(model, train_loader, optimizer, scheduler, device, epochs=100)

# Evaluate the model
evaluate_model(model, test_loader, device)
