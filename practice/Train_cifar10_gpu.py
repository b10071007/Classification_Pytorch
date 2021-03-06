# -*- coding: utf-8 -*-

import os
from time import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

#--------------------------------------------------------------------------------------------------------#

def GetCurrentTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#--------------------------------------------------------------------------------------------------------#

img_path = "D:/Dataset/Classification"
model_path = "D:/Coding/pytorch/project/Classification_Pytorch/model/"
model_dir = model_path + 'cifar_net_gpu.pth'

max_epoch = 2
display_interval = 50


# 1. Loading and normalizing CIFAR10


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root=img_path, train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root=img_path, train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# ------------------------------------------------------------------------------#

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

net = Net()

# Training on GPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
# ------------------------------------------------------------------------------#

# 3. Define a Loss function and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ------------------------------------------------------------------------------#

# 4. Train the network
# num_iterations = 0
batch_time = []

# start = torch.cuda.Event(enable_timing=True)
# end = torch.cuda.Event(enable_timing=True)

log_str = "Epoch: [{:3d}/{:3d}] Iterations: {:4d} Loss: {:.3f} Batch_time: {:.2f} ms"

print('{}: Start Training '.format(GetCurrentTime()))
for epoch in range(max_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        
        # start.record()
        torch.cuda.synchronize()
        start = time()*1000

        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # end.record()
        # torch.cuda.synchronize()
        torch.cuda.synchronize()
        end = time()*1000

        # num_iterations += 1
        # batch_time.append(start.elapsed_time(end))
        batch_time.append(end-start)
        
        # print statistics
        running_loss += loss.item()
        if i % display_interval == (display_interval - 1):  # print every 'display_interval' mini-batches
            print(log_str.format(epoch, max_epoch, i + 1, 
                                 running_loss / display_interval, 
                                 np.mean(batch_time)))
            batch_time = []
            # print('[%d, %5d] loss: %.3f' %
            #       (epoch + 1, i + 1, running_loss / display_interval))
            running_loss = 0.0

print('{}: Training Finished  '.format(GetCurrentTime()))

os.makedirs(model_path, exist_ok=True)
torch.save(net.state_dict(), model_dir)

# ------------------------------------------------------------------------------#

# 5. Test the network on the test data

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        # images, labels = data
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

# ------------------------------------------------------------------------------#