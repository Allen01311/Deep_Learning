import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets, transforms, models
from tqdm import tqdm
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.ImageFolder(root='D:/image_experience/hw2/hw2train', transform=transform)
test_dataset = datasets.ImageFolder(root='D:/image_experience/hw2/hw2test', transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

#用GPU跑
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {device}")

#--
#前置:Model權重設定
#AlexNet
alexnet_model = models.alexnet(weights="AlexNet_Weights.DEFAULT").to(device)
alexnet_model.classifier[6] = nn.Linear(4096, 2).to(device)

#ResNet
resnet_model = models.resnet18(weights="ResNet18_Weights.DEFAULT").to(device)
resnet_model.fc = nn.Linear(512, 2).to(device)

criterion = nn.CrossEntropyLoss().to(device)
alexnet_optimizer = optim.SGD(alexnet_model.parameters(), lr=0.00005, momentum=0.9)
resnet_optimizer = optim.SGD(resnet_model.parameters(), lr=0.00005, momentum=0.9)

#--

#訓練過程
def train_and_validate(model, optimizer, criterion, train_loader, test_loader, num_epochs=5):
    training_loss = []
    validation_loss = []
    print('Begin training')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i ,(inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'Loss': loss.item()})
        
        average_loss = running_loss / len(train_loader)
        training_loss.append(average_loss)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {average_loss}')

    #計算accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy}')

    return training_loss


#繪圖(training_loss、validation_loss)
def plot_validation_loss_curve(training_loss, title, figure_name):
    plt.plot(training_loss, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.show()
    plt.savefig(figure_name)

alexnet_training_loss= train_and_validate(alexnet_model, alexnet_optimizer, criterion, train_loader, test_loader)
print('AlexNet has finished')
torch.save(alexnet_model.state_dict(), 'alexnet_model_weights.pth')
plot_validation_loss_curve(alexnet_training_loss, 'AlexNet Training Loss','AlexNet_Figure')

resnet_training_loss = train_and_validate(resnet_model, resnet_optimizer, criterion, train_loader, test_loader)
print('ResNet has finished')
torch.save(resnet_model.state_dict(), 'resnet_model_weights.pth')
plot_validation_loss_curve(resnet_training_loss, 'ResNet Training Loss','ResNet_Figure')