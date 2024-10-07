import torch
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = outputs.cpu().numpy()
            
            # Visualize first image and prediction
            plt.subplot(1, 3, 1)
            plt.imshow(images[0].cpu().permute(1, 2, 0))
            plt.title('Original Image')
            
            plt.subplot(1, 3, 2)
            plt.imshow(masks[0].cpu().squeeze(), cmap='gray')
            plt.title('Ground Truth Mask')
            
            plt.subplot(1, 3, 3)
            plt.imshow(outputs[0].squeeze(), cmap='gray')
            plt.title('Predicted Mask')
            
            plt.show()
            break
