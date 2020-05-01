import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import classifyDataset as cDataset

#--------------------------------------------------------------------------------------------------------#

rootPath = "D:/Dataset/Classification/cifar10/"
imgPath = rootPath + "_Images/"
train_fListPath = rootPath + "train.txt"
val_fListPath = rootPath + "val.txt"

model_path = "D:/Coding/pytorch/experiment/custom_cifar10/model/"
model_dir = model_path + 'cifar_net_gpu.pth'

batch_size_train = 200
batch_size_val = 10
num_epochs = 2
display_interval = 20

# 1. Loading and normalizing Custom cifar10 dataset
print("Setup dataset ...")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_Dataset = cDataset.ClassifyDataset(train_fListPath, imgPath, transform=transform)
train_Loader = DataLoader(train_Dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
val_Dataset = cDataset.ClassifyDataset(val_fListPath, imgPath, transform=transform)
val_Loader = DataLoader(train_Dataset, batch_size=batch_size_val, shuffle=False, num_workers=0)

num_train = len(train_Dataset)
epoch_size = num_train // batch_size_train

#--------------------------------------------------------------------------------------------------------#

# 2. Define a Convolutional Neural Network

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        input_shape = np.array(x.shape[1:])
        x = x.view(-1, input_shape.prod())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

print("Create Model ...")
net = Net()

# Training on GPU
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(" - gpu is available -> use gpu")
else:
    device = torch.device("cpu")
    print(" - gpu is not available -> use cpu")

net.to(device)

#--------------------------------------------------------------------------------------------------------#

# 3. Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#--------------------------------------------------------------------------------------------------------#

# 4. Train the network

print("Start to train custom cifar10 dataset ...")
for epoch in range(num_epochs):

    running_loss = 0.0
    for i, data in enumerate(train_Loader, 0):
        # get the inputs; data is a dictionary of {image: X, label: X}
        inputs = data['image']
        labels = data['label']
        inputs, labels = data['image'].to(device), data['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % display_interval == (display_interval - 1):  # print every 'display_interval' mini-batches
            # print('[%d, %5d] loss: %.3f' %
            #       (epoch + 1, i + 1, running_loss / display_interval))
            print( "epoch:[%d/%d] iter:[%d/%d]  loss: %.3f" %
                   (epoch + 1, num_epochs, i+1, epoch_size, running_loss / display_interval) )
            running_loss = 0.0

print('Finished Training')

os.makedirs(model_path, exist_ok=True)
torch.save(net.state_dict(), model_dir)

# ------------------------------------------------------------------------------#

# 5. Test the network on the test data

correct = 0
total = 0
with torch.no_grad():
    for data in val_Loader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print( 'Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total) )

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in batch_size_val:
        inputs = data['image']
        labels = data['label']
        inputs, labels = data['image'].to(device), data['label'].to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print( 'Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]) )

# ------------------------------------------------------------------------------#
