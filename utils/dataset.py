import os
import cv2
import numpy as np
from torch.utils.data import Dataset


def preprocess_mask(mask):
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask.astype(np.float32) / 255


class KeyholeDataset(Dataset):
    def __init__(self, dataset_directory, transform=None):
        self.dataset_directory = dataset_directory
        self.image_directory = os.path.join(dataset_directory, 'JPEGImages')
        self.mask_directory = os.path.join(dataset_directory, 'SegmentationClass')
        self.image_filenames = os.listdir(self.image_directory)
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image = cv2.imread(os.path.join(self.image_directory, image_filename))
        mask = cv2.imread(os.path.join(self.mask_directory, image_filename), cv2.IMREAD_GRAYSCALE)
        mask = preprocess_mask(mask)

        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask
