import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

def train_model(model, train_loader, optimizer, scheduler, device, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0
        running_iou = 0.0
        
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_coefficient_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_dice += dice_coefficient(outputs, masks).item()
            running_iou += iou(outputs, masks).item()

        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        avg_dice = running_dice / len(train_loader)
        avg_iou = running_iou / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}")

