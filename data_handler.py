import torch
import torch.nn as nn
import torch.nn.functional as F

import lightning as pl

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import cv2
import numpy as np
import h5py

import matplotlib.pyplot as plt

class cellDataset(Dataset):
    def __init__(self, images_list, transform=None):
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)
    
    def __getitem__(self, idx):
        img_path = self.images_list[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))

        gt_path = img_path.replace("images", "densities").replace(".png", ".h5")
        gt_file = h5py.File(gt_path, 'r')
        gt = np.asarray(gt_file['density'])

        gt = cv2.resize(gt, (img.shape[1]//16, img.shape[0]//16),interpolation= cv2.INTER_AREA) * 16 * 16
        gt = gt.astype(np.float32)

        # Convert to tensor
        img = transforms.ToTensor()(img)
        gt = torch.from_numpy(gt).unsqueeze(0)

        return img, gt
    

# # Test the dataset

# from glob import glob

# images_list = glob("../Datasets/DCC/trainval/images/*.png")

# print(f"Total images: {len(images_list)}")

# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip()
# ])

# dataset = cellDataset(images_list, transform=transform)

# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# for img, gt in dataloader:
#     plt.figure(figsize=(10, 10))
#     plt.subplot(1, 2, 1)
#     plt.imshow(img[0].permute(1, 2, 0))
#     plt.title("Image")
#     plt.axis("off")
#     plt.subplot(1, 2, 2)
#     plt.imshow(gt[0].squeeze())
#     plt.title(f"GT count: {gt[0].sum()}")
#     plt.axis("off")
#     plt.show()

#     break
    



