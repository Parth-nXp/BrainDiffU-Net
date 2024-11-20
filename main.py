import torch
import pandas as pd
from glob import glob
from torch.utils.data import DataLoader
from torch import optim
from torchvision import transforms
from utils import set_seed, initialize_device
from dataset import BrainMRIDataset
from model import UNet
from training import train_node_model
from data_utils import split_data_into_nodes

if __name__ == "__main__":
    set_seed()
    device = initialize_device()

    # Parameters
    num_nodes = 15
    radius = 0.3
    batch_size = 16
    learning_rate = 1e-3
    total_iterations = 150
    epochs_per_iteration = 5
    image_h, image_w = 256, 256

    # Load and preprocess data
    mask_images = glob(r'./Dataset/kaggle_3m/*/*_mask*')
    image_filenames = [i.replace("_mask", "") for i in mask_images]
    chunks_images, chunks_masks = split_data_into_nodes(image_filenames, mask_images, num_nodes)

    # Convert chunks into dataframes
    dfs = [
        pd.DataFrame({'image_filename': chunks_images[i], 'mask_images': chunks_masks[i]})
        for i in range(len(chunks_images))
    ]

    # Initialize models, datasets, loaders, optimizers, and schedulers
    models = [UNet() for _ in range(num_nodes)]

    # Wrap models with nn.DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        models = [torch.nn.DataParallel(model).to(device) for model in models]
    else:
        print("Only one GPU available. Using single GPU.")
        models = [model.to(device) for model in models]

    # Data transforms
    train_transforms = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop((image_h, image_w), scale=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # Create datasets and DataLoaders
    datasets = [BrainMRIDataset(dfs[i], image_transform=train_transforms, mask_transform=train_transforms) for i in range(num_nodes)]
    loaders = [DataLoader(datasets[i], batch_size=batch_size, shuffle=True) for i in range(num_nodes)]

    # Optimizers and schedulers
    optimizers = [optim.Adam(models[i].parameters(), lr=learning_rate) for i in range(num_nodes)]
    schedulers = [optim.lr_scheduler.StepLR(optimizers[i], step_size=30, gamma=0.1) for i in range(num_nodes)]

    # Training loop
    for iteration in range(total_iterations):
        print(f"\n===== Iteration {iteration + 1}/{total_iterations} =====")
        for epoch in range(epochs_per_iteration):
            print(f"-- Epoch {epoch + 1}/{epochs_per_iteration} --")
            for node_id, model in enumerate(models):
                train_node_model(node_id, model, loaders[node_id], optimizers[node_id], schedulers[node_id], device)

    print("Training completed.")
