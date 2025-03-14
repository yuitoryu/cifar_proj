import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import json
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os

from model.res3net_k533 import Res3NetK533, BasicBlock3

# Code is made based on the file provided in Kaggle competition and help of ChatGPT and DeepSeek R1

def model_testing(model, device, test_dataloader, test_acc, test_losses, misclassified = []):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        for index, (data, target) in enumerate(test_dataloader):
            data, target = data.to(device), target.to(device)
            with autocast(): # Mixed precision
            
                output = model(data)
            pred = output.argmax(dim=1, keepdim=True)

            for d,i,j in zip(data, pred, target):
                if i != j:
                    misclassified.append([d.cpu(),i[0].cpu(),j.cpu()]) # Counting correct and incorrect answers

            test_loss += F.nll_loss(output, target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_dataloader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)))

    test_acc.append(100. * correct / len(test_dataloader.dataset))
    return misclassified

def model_training(model, device, train_dataloader, optimizer, train_acc, train_losses, criterion):
    model.train()
    correct = 0
    processed = 0
    scaler = GradScaler(enabled=True)  # Enable scaler
    running_loss = 0.0
    PBAR = tqdm(train_dataloader)

    for batch_idx, (data, target) in enumerate(PBAR): # Loop over batches
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        with autocast(enabled=True):
            y_pred = model(data)
            loss = criterion(y_pred, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        # print statistics
        running_loss += loss.item()
        train_acc.append(100*correct/processed)

    
    print(f'Loss={loss.item():.4f} lr={optimizer.param_groups[0]["lr"]} Accuracy={100*correct/processed:0.2f}')
    train_losses.append(loss.item())
    
def lr_warmup(current_epoch):
    if current_epoch < 5:  # Warmup for 5 epochs
        return (0.01 + (0.1 - 0.01) * (current_epoch / 5))
    else:
        return 1.0  # Use CosineAnnealingLR

# Convert to TensorDataset and apply transformations
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.transform(self.images[idx])  # Apply data augmentation
        label = self.labels[idx]
        return img, label
    
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
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    
    batch_test_dict = load_cifar_batch(os.path.join(cifar10_dir, 'test_batch'))
    val_images = batch_test_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    val_labels = np.array(batch_test_dict[b'labels'])
    
    val_dataset = CustomCIFAR10Dataset(val_images, val_labels, transform=test_transform)
    
    # DataLoaders
    # If you want to run this code, make sure to change to these parameters depend on your device
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=12,
                              prefetch_factor=4, persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4, 
                              persistent_workers=True, pin_memory=True)
    
    # Load test dataset
    cifar_test_path = 'deep-learning-spring-2025-project-1/cifar_test_nolabel.pkl'
    test_batch = load_cifar_batch(cifar_test_path)
    test_images = test_batch[b'data'].astype(np.float32) / 255.0
    
    # Convert test dataset to Tensor
    test_dataset = [(test_transform(img),) for img in test_images]
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    #train_dataset[0][0] == transform(train_data[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Importing Model and printing Summary
    model = Res3NetK533(BasicBlock3).to(device)
    
    cwd = os.getcwd()

    criterion = nn.NLLLoss()

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    
    # Set up warmup
    warmup_epochs = 5
    EPOCHS = 200

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
    
    train_acc = []
    train_losses = []
    valid_acc = []
    valid_losses = []
    
    
    
    for epoch in range(EPOCHS):
        print(f'EPOCH : {epoch}')
        
        # Manually adjust LR
        if epoch < warmup_epochs:
            lr = 0.01 + (0.1 - 0.01) * (epoch / warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        
        # Training
        model_training(model, device, train_loader, optimizer, train_acc, train_losses, criterion)

        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Validation
        misclassified = model_testing(model, device, valid_loader, valid_acc, valid_losses)
        
    torch.save(model.state_dict(), "res3netk533/model_checkpoint.pth")
    stats = {'train_acc': train_acc,
             'train_losses': train_losses,
             'valid_acc': valid_acc,
             'valid_losses': valid_losses}                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    with open("res3netk533/stats2.json", "w") as file:
        json.dump(stats, file, indent=4)  # `indent=4` makes it readable
    
if __name__ == "__main__":
    main()
