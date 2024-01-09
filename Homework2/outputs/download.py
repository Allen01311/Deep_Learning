import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
import gzip
import numpy as np

# 下載 MNIST dataset
full_dataset = datasets. MNIST (root='data', train=True, download=True)
class MNISTDataset (Dataset):
    def __init__(self, data_file, label_file, transform=None):
        self.transform = transform
        #讀取圖像資料
        with gzip.open(data_file, 'rb') as f:
            self.images = np.frombuffer (f.read(), np. uint8, offset=16).reshape(-1, 28, 28)
        #讀取標籤資料
        with gzip.open(label_file, 'rb') as f:
            self.labels = np.frombuffer (f.read(), np. uint8, offset=8)
    #查看照片總張數
    def _len__(self):
        return len(self.images)
    
    def getitem_(self, idx): 
        image = self.images[idx]
        label = self.labels[idx]
        image = np.reshape(image, (28, 28)) #28x28 
        image = np.resize(image, (224, 224)) #224x224
        if self.transform:
            image = self.transform(image)
            
        return image, label
