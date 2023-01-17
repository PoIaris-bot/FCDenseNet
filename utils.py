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


def augment_hsv(image, h_gain=0.5, s_gain=0.5, v_gain=0.5):
    r = np.random.uniform(-1, 1, 3) * [h_gain, s_gain, v_gain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    dtype = image.dtype

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    image_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)
