import os
import cv2
import sys
import platform
from pathlib import Path
from torch.utils.data import Dataset

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.transform import resize, transform


class KeyholeDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.image_names = os.listdir(os.path.join(data_path, 'JPEGImages'))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image_path = os.path.join(self.data_path, 'JPEGImages', image_name)
        segment_image_path = os.path.join(self.data_path, 'SegmentationClass', image_name)

        _, binary = cv2.threshold(cv2.imread(segment_image_path, 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        segment_image = resize(binary)
        image = resize(cv2.imread(image_path))
        return transform(image), transform(segment_image)