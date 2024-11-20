import cv2
from PIL import Image
from torch.utils.data import Dataset

class BrainMRIDataset(Dataset):
    """
    Custom dataset for Brain MRI images and masks.

    Parameters:
    dataframe (pandas.DataFrame): DataFrame containing file paths for images and masks.
    image_transform (torchvision.transforms.Compose): Transformations for the images.
    mask_transform (torchvision.transforms.Compose): Transformations for the masks.
    target_size (tuple): Target size for resizing images and masks.
    """
    def __init__(self, dataframe, image_transform=None, mask_transform=None, target_size=(256, 256)):
        self.dataframe = dataframe
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.target_size = target_size

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx]['image_filename']
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_path = self.dataframe.iloc[idx]['mask_images']
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, self.target_size)
        mask = cv2.resize(mask, self.target_size)

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        mask = mask.unsqueeze(0)
        return image, mask
