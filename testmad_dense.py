import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from madgrad.madgrad import MADGRAD
import matplotlib.pyplot as plt

# Transform the data (mean/std for CIFAR-10 RGB)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define a CNN model for CIFAR-10
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # CIFAR-10 has 3 channels
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Adjusted input size after conv and pooling
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

def train_model(optimizer, model):
    criterion = nn.CrossEntropyLoss()
    losses = []

    for epoch in range(30):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch+1}, {i+1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
        losses.append(running_loss)
    return losses

# Instantiate and train models with different optimizers
model_madgrad = Net()
model_adam = Net()
model_sgd = Net()

optimizer_madgrad = MADGRAD(model_madgrad.parameters(), lr=0.001, momentum=0.9, weight_decay=0)
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.001, momentum=0.9)

losses_madgrad = train_model(optimizer_madgrad, model_madgrad)
losses_adam = train_model(optimizer_adam, model_adam)
losses_sgd = train_model(optimizer_sgd, model_sgd)
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total
acc_madgrad = evaluate_model(model_madgrad, testloader)
acc_adam = evaluate_model(model_adam, testloader)
acc_sgd = evaluate_model(model_sgd, testloader)

print(f"MADGRAD Test Accuracy: {acc_madgrad:.2%}")
print(f"Adam Test Accuracy: {acc_adam:.2%}")
print(f"SGD Test Accuracy: {acc_sgd:.2%}")


# Plot losses
epochs = range(1, 31)
plt.plot(epochs, losses_madgrad, label='MADGRAD')
plt.plot(epochs, losses_adam, label='Adam')
plt.plot(epochs, losses_sgd, label='SGD')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison on CIFAR-10')
plt.legend()
plt.show()
