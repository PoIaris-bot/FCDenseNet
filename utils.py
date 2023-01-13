import cv2
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


def resize(image, size=256):
    height, width = image.shape[:2]
    max_size = max(height, width)

    if len(image.shape) == 3:
        image_new = np.zeros((max_size, max_size, 3), np.uint8)
        image_new[:height, :width, :] = image
    else:
        image_new = np.zeros((max_size, max_size), np.uint8)
        image_new[:height, :width] = image
    image_new = cv2.resize(image_new, (size, size))
    return image_new
