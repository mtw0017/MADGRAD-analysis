import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from madgrad.madgrad import MADGRAD

# Transform the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define your model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

def train_model(optimizer, model):
    # Define your loss function
    criterion = nn.CrossEntropyLoss()

    losses = []
    # Train your model
    for epoch in range(5):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        losses.append(running_loss)
    return losses

# Instantiate your model
model_madgrad = Net()
model_adam = Net()
model_sgd = Net()

# Define your MADGRAD optimizer
optimizer_madgrad = MADGRAD(params=model_madgrad.parameters(), lr=0.001, momentum=0.9, weight_decay=0)


# Define your ADAM optimizer
optimizer_adam = optim.Adam(model_adam.parameters(), lr=0.001)

# Define your SGD optimizer
optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.001, momentum=0.9)

# Train models with different optimizers
losses_madgrad = train_model(optimizer_madgrad, model_madgrad)
losses_adam = train_model(optimizer_adam, model_adam)
losses_sgd = train_model(optimizer_sgd, model_sgd)

# Plot the losses
import matplotlib.pyplot as plt

epochs = range(1, 6)
plt.plot(epochs, losses_madgrad, label='MADGRAD')
plt.plot(epochs, losses_adam, label='Adam')
plt.plot(epochs, losses_sgd, label='SGD')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.show()

