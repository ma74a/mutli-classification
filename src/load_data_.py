from torchvision import transforms
from src.custom_dataset import CustomDataset
from torch.utils.data import DataLoader, random_split
from utils.config import *

transform = {
    "train": transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        ]),

    "test": transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),

    ])
}

def load_data():
    global transform

    train_dataset = CustomDataset("../data/train", transform["train"])
    val_dataset = CustomDataset("../data/val", transform["test"])
    test_dataset = CustomDataset("../data/test", tranform=transform["test"])

    class_to_idx = train_dataset.class_to_idx

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, class_to_idx