import os
import random

import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

from tqdm import tqdm

class MRSSCDataset(Dataset):
    def __init__(self, root_dir, cls_map, X, y, transform=None):
        self.root_dir = root_dir
        self.cls_map = cls_map
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image = Image.open(os.path.join(self.root_dir, self.y[idx], self.X[idx]))
        # image = Image.open(self.X[idx])
        label = self.cls_map.get(self.y[idx])

        target_size = (512, 512)
        if image.size != target_size:
            image = image.resize(target_size)

        if self.transform:
            image = self.transform(image)

        return image, label
    
def _load_data(root_dir):
    X = []
    y = []
    cls_map = {}
    cls_count = {}
    data = {}

    for target in ["Target1", "Target2", "Target3"]:
        classes = os.listdir(os.path.join(root_dir, target))
        for i, cls in enumerate(classes):
            if cls not in cls_map.keys():
                cls_map.update({cls: i})
                cls_count.update({cls: 0})
                data.update({cls: []})
            cls_dir = os.path.join(root_dir, target, cls)
            for file in tqdm(os.listdir(cls_dir)):
                if np.array(Image.open(os.path.join(cls_dir, file))).shape == (256,256):
                    data[cls].append(os.path.join(cls_dir, file))
                    cls_count[cls] += 1
    
    X = []
    y = []

    for item in data.items():
        cls, path = item
        if len(path) > 1000:
            path = random.choices(path,k=1000)
    
        X.extend(path)
        y.extend([cls]*len(path))

    return cls_map, X, y
    
def _create_dataloader(root_dir, batch_size, num_workers=0):
    classes = os.listdir(root_dir)
    X = []
    y = []
    cls_map = {}
    for i, cls in tqdm(enumerate(classes)):
        cls_map.update({cls: i})
        cls_dir = os.path.join(root_dir, cls)
        # images = random.choices(os.listdir(cls_dir),k=600)
        images = []
        for file in os.listdir(cls_dir):
            if np.array(Image.open(os.path.join(cls_dir, file))).shape == (256,256,3):
                images.append(file)
        X.extend(images)
        y.extend([cls]*len(images))

    # cls_map, X, y = _load_data(root_dir)

    X_train, X_test, y_train, y_test  = train_test_split(X,y,stratify=y, test_size=0.1, random_state=42)

    # transform = v2.Compose(
    #     [v2.ToTensor(),
    #      v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    transform = v2.Compose(
        [v2.ToTensor(), v2.GaussianBlur(5, 1)])
    
    training_set = MRSSCDataset(root_dir, cls_map, X_train, y_train, transform=transform)
    validation_set = MRSSCDataset(root_dir, cls_map, X_test, y_test, transform=transform)

    training_loader = iter(DataLoader(training_set, batch_size, shuffle=True, num_workers=num_workers))
    validation_loader = iter(DataLoader(validation_set, batch_size, shuffle=True,num_workers=num_workers))

    return cls_map, training_loader, validation_loader

if __name__ == "__main__":
    cls_map, training_loader, validation_loader = _create_dataloader("./data/earth/source", 128)
    print(cls_map)
    print(next(training_loader)[0].size())
