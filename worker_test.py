# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 01:53:39 2025

@author: seer2
"""

import torch
from torch.optim.lr_scheduler import LambdaLR
from torch import optim
import os
from torch import nn
import torch.nn.functional as F 
import pickle
from torchsummary import summary
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from tqdm import tqdm
from model.resnet import ResNet, BasicBlock
import json
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split, TensorDataset
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler
import os
import time

# Convert to TensorDataset and apply transformations
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)  # 保持为numpy数组
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img = Image.fromarray(self.images[idx])  # 直接转PIL图像（更高效）
        img = self.transform(self.images[idx])  # 应用transform
        label = self.labels[idx]
        return img, label

def test_num_workers(dataset, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    start_time = time.time()
    for _ in loader:
        pass  # 仅测试数据加载速度，不进行训练
    end_time = time.time()
    
    return end_time - start_time

def main():
    # auto. choose CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Function to load CIFAR-10 dataset
    def load_cifar_batch(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
    
    # Specify the directory containing CIFAR-10 batches
    cifar10_dir = 'deep-learning-spring-2025-project-1/cifar-10-python/cifar-10-batches-py'
    
    # Load metadata (labels)
    meta_data_dict = load_cifar_batch(os.path.join(cifar10_dir, 'batches.meta'))
    label_names = [label.decode('utf-8') for label in meta_data_dict[b'label_names']]
    
    # Load training data
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = load_cifar_batch(os.path.join(cifar10_dir, f'data_batch_{i}'))
        train_data.append(batch[b'data'])
        train_labels += batch[b'labels']
    
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # Convert to HWC format
    train_labels = np.array(train_labels)
    
    # Data augmentation and normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 随机擦除增强
    ])
    
    
    
    train_dataset = CustomCIFAR10Dataset(train_data, train_labels, transform=transform)
    
    num_workers_list = [1, 2, 4, 6, 8, 10, 12, 16]  # 测试的num_workers值
    results = {}
    
    for num_workers in num_workers_list:
        duration = test_num_workers(train_dataset, num_workers)
        results[num_workers] = duration
        print(f'num_workers={num_workers}, Time={duration:.2f}s')
        
if __name__ == "__main__":
    main()

    