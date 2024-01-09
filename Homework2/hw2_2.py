import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt

#下載資料集
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transform)

#用GPU跑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

#CNN模型設置
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#VGG模型設置
# class VGGModel(nn.Module):
#     def __init__(self):
#         super(VGGModel, self).__init__()
#         self.vgg11 = models.vgg11(weights='VGG11_Weights.DEFAULT').to(device)
#         self.vgg11.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)

#     def forward(self, x):
#         return self.vgg11(x)

#VGG模型設置
class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(-1, 256)
        return x

#------------------------------------------------------------------------------

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

#把models、optimizers、loss function都用GPU跑
cnn_model = CNNModel().to(device)
vgg_model = VGGModel().to(device)
cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)
vgg_optimizer = optim.Adam(vgg_model.parameters(), lr=0.001)

criterion = nn.CrossEntropyLoss().to(device)

#Training 的Function
def train(model, criterion, optimizer, train_loader, epochs=3):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{epochs}')
        for i ,(inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            labels = labels.squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})

        average_loss = running_loss / len(train_loader)
        train_losses.append(average_loss)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}')

    return train_losses

#計算accuracy
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

#------------------------------------------------------------------------------

#訓練CNN
cnn_train_losses = train(cnn_model, criterion, cnn_optimizer, train_loader)

#計算CNN accuracy
cnn_accuracy = evaluate(cnn_model, test_loader)
print(f"CNN Test Accuracy: {cnn_accuracy}")

#------------------------------------------------------------------------------

#訓練 VGG
vgg_train_losses = train(vgg_model, criterion, vgg_optimizer, train_loader)

#計算 VGG accuracy
vgg_accuracy = evaluate(vgg_model, test_loader)
print(f"VGG Test Accuracy: {vgg_accuracy}")

#繪製training losses
plt.plot(cnn_train_losses, label='CNN Training Loss')
plt.plot(vgg_train_losses, label='VGG Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()