from torchvision import transforms
from torch.utils.data import Dataset

import os
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, data_dir, tranform=None):
        self.data_dir = data_dir
        self.transorm = tranform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self._load_data()

    def _load_data(self):
        """load the dataset from the directory"""

        for class_idx, class_name in enumerate(sorted(os.listdir(self.data_dir))):
            class_dir = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = class_idx
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)

    def __len__(self):
        """return the size of the data"""

        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """return tuple (image, label)"""

        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transorm:
            img = self.transorm(img)
        
        return img, label
    
