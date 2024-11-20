import torch
import torch.nn as nn  # Add this import
from metrics import dice_coefficient, iou

def train_node_model(node_id, model, train_loader, optimizer, scheduler, device):
    """
    Train a model for a specific node.

    Parameters:
    node_id (int): Node identifier.
    model (torch.nn.Module): Model to train.
    train_loader (DataLoader): DataLoader for training data.
    optimizer (torch.optim.Optimizer): Optimizer for training.
    scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
    device (torch.device): Device to train the model on.

    Returns:
    tuple: Average loss, Dice coefficient, and IoU score.
    """
    print(f"Training model for Node {node_id}.")
    model.train()
    running_loss, running_dice, running_iou = 0.0, 0.0, 0.0
    criterion = nn.BCELoss()  # Ensure this works with the proper import

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Ensure masks have the correct shape
        masks = masks.squeeze(1)  # Remove extra dimension if present

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_dice += dice_coefficient(outputs, masks).item()
        running_iou += iou(outputs, masks).item()

    avg_loss = running_loss / len(train_loader)
    avg_dice = running_dice / len(train_loader)
    avg_iou = running_iou / len(train_loader)

    scheduler.step()
    print(f"Node {node_id} - Avg Loss: {avg_loss:.4f}, Avg Dice: {avg_dice:.4f}, Avg IoU: {avg_iou:.4f}")
    return avg_loss, avg_dice, avg_iou
