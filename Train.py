
import os
import numpy as np
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append("E:/Coding/pytorch/project/Classification_Pytorch/")
from dataset import classifyDataset as cDataset
from models import VGG

#--------------------------------------------------------------------------------------------------------#

# setup output manager
class OutputManager():
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        filePath = os.path.split(log_file_path)[0]
        if not os.path.exists(filePath): os.makedirs(filePath)
        self.log_file = open(log_file_path, 'w')

    def output(self, content):
        print(content)
        self.log_file.write(content + '\n')

    def close(self):
        self.log_file.close()

#--------------------------------------------------------------------------------------------------------#
        
def GetCurrentTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def val(net, val_Loader, device, outputManage):
    outputManage.output("Start validation")
    val_display_interval = 100
    correct = 0
    total = 0
    with torch.no_grad():
        for iter_val, data in enumerate(val_Loader):
            if iter_val % val_display_interval == (val_display_interval - 1):
                outputManage.output("Process: {}".format(iter_val+1))

            # images, labels = data
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = (100 * correct / total)
    outputManage.output( 'Accuracy on validation images: %d %%' % accuracy )
    
    return accuracy
    # class_correct = list(0. for i in range(10))
    # class_total = list(0. for i in range(10))
    # with torch.no_grad():
    #     for data in val_Loader:
    #         images, labels = data[0].to(device), data[1].to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs, 1)
    #         c = (predicted == labels).squeeze()
    #         for i in range(4):
    #             label = labels[i]
    #             class_correct[label] += c[i].item()
    #             class_total[label] += 1
    
    # for i in range(10):
    #     print( 'Accuracy of %5s : %2d %%' % (i, 100 * class_correct[i] / class_total[i]) )


def train(net, train_Loader, val_Loader, device, max_epoch, display_interval, epoch_size, val_interval, outputManage, best_model_path):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    
    batch_time = []
    log_str = "Epoch: [{:3d}/{:3d}] Iterations: {:4d} Loss: {:.3f} Batch_time: {:.2f} ms"

    print('{}: Start Training '.format(GetCurrentTime()))
    best_ep = 0
    best_acc = 0
    best_model_message = "Best model until now:\n - epoch={}\n - Accuracy={}"

    for epoch in range(max_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        for iter, data in enumerate(train_Loader):
            
            # start.record()
            torch.cuda.synchronize()
            start = time.time()*1000

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
            end = time.time()*1000

            # num_iterations += 1
            # batch_time.append(start.elapsed_time(end))
            batch_time.append(end-start)

            # print statistics
            running_loss += loss.item()
            if iter % display_interval == (display_interval - 1):  # print every 'display_interval' mini-batches
                outputManage.output(log_str.format(epoch, max_epoch, iter + 1, 
                                    running_loss / display_interval, 
                                    np.mean(batch_time)))
                batch_time = []
                running_loss = 0.0

        # Validation
        if epoch % val_interval == (val_interval - 1):
            accuracy = val(net, val_Loader, device, outputManage)
            if accuracy > best_acc:
                best_acc = accuracy
                best_ep = epoch
                torch.save(net.state_dict(), best_model_path)

            outputManage.output(best_model_message.format(best_ep, best_acc) + '\n')


    outputManage.output('{}: Training Finished  '.format(GetCurrentTime()))
    best_acc = round(best_acc, 4)
    with open(os.path.split(best_model_path)[0] + '/Best_model_{}_{}.txt'.format(best_ep, best_acc), 'w') as fObj:
        fObj.writelines("{}\t{}".format(best_ep, best_acc))


def main():
    
    rootPath = "E:/Dataset/Classification/cifar10/"
    imgPath = rootPath + "_Images/"
    train_fListPath = rootPath + "train.txt"
    val_fListPath = rootPath + "val.txt"

    model_name = "VGG16"
    save_folder = "E:/Coding/pytorch/project/Classification_Pytorch/weights/"
    best_model_path = os.path.join(save_folder, model_name + "_Best.pth")

    num_classes = 10
    batch_size_train = 400
    batch_size_val = 100
    max_epoch = 100
    display_interval = 25
    val_interval = 10

    #--------------------------------------------------------------------------------------------------------#

    # Setup output information 
    os.makedirs(save_folder, exist_ok=True)
    log_file_path = os.path.join(save_folder, model_name + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log')
    
    outputManage = OutputManager(log_file_path)

    # Loading and normalizing Custom cifar10 dataset
    outputManage.output("Setup dataset ...")
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_Dataset = cDataset.ClassifyDataset(train_fListPath, imgPath, transform=transform)
    train_Loader = DataLoader(train_Dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
    val_Dataset = cDataset.ClassifyDataset(val_fListPath, imgPath, transform=transform)
    val_Loader = DataLoader(val_Dataset, batch_size=batch_size_val, shuffle=False, num_workers=0)

    num_train = len(train_Dataset)
    epoch_size = num_train // batch_size_train
    
    # Setup model
    outputManage.output("Create Model ...")
    if model_name=="VGG16":
        net = VGG.VGG16(num_classes, init_weights=True)
    elif model_name=="VGG19":
        net = VGG.VGG19(num_classes, init_weights=True)
    else:
        raise ValueError("Not support \"{}\"".format(model_name))


    # Setup GPU
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        outputManage.output(" - GPU is available -> use GPU")
    else:
        device = torch.device("cpu")
        outputManage.output(" - GPU is not available -> use GPU")

    net.to(device)
    
    train(net, train_Loader, val_Loader, device, max_epoch, display_interval, epoch_size, val_interval, outputManage, best_model_path)

#--------------------------------------------------------------------------------------------------------#  
    
if __name__ == '__main__':
    main()