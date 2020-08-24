
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
from models import VGG, ResNet

#--------------------------------------------------------------------------------------------------------#

# Setup settings and hyper-parameter
class Setting():
    def __init__(self, num_classes, batch_size_train, batch_size_val, max_epoch, display_interval, val_interval,
                 base_lr, gamma, lr_decay_steps, warm_epoch):
        self.num_classes = num_classes
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.max_epoch = max_epoch
        self.display_interval = display_interval
        self.val_interval = val_interval
        self.base_lr = base_lr
        self.gamma = gamma
        self.lr_decay_steps = lr_decay_steps
        self.warm_epoch = warm_epoch

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

def adjust_learning_rate(optimizer, base_lr, gamma, step_index, 
                         epoch, warm_epoch, iteration, epoch_iters):
    if epoch < warm_epoch: # warm up
        total_iteration = epoch_iters*epoch + iteration
        lr = 1e-6 + (base_lr - 1e-6) * total_iteration / (epoch_iters * warm_epoch)
    else:
        lr = base_lr * (gamma ** (step_index))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

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


def train(net, train_Loader, val_Loader, device, setting, epoch_iters, outputManage, best_model_path):
    
    # Setup output messages
    batch_time = []
    log_str = "Epoch: [{:3d}/{:3d}] Iterations: [{:3d}/{:3d}] Loss: {:.3f} Batch_time: {:.2f} ms LR: {:.4f}"

    # Setup Best model saving
    print('{}: Start Training '.format(GetCurrentTime()))
    best_ep = 0
    best_acc = 0
    best_model_message = "Best model until now:\n - epoch={}\n - Accuracy={}"

    # Setup optimizer
    step_index = 0
    lr = setting.base_lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)

    for epoch in range(setting.max_epoch):  # loop over the dataset multiple times
        
        if (epoch+1) in setting.lr_decay_steps:
            step_index += 1

        running_loss = 0.0
        for iteration, data in enumerate(train_Loader):
            
            lr = adjust_learning_rate(optimizer, setting.base_lr, setting.gamma, step_index, 
                                      epoch, setting.warm_epoch, iteration, epoch_iters)

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
            if iteration % setting.display_interval == (setting.display_interval - 1):  # print every 'display_interval' mini-batches
                outputManage.output(
                    log_str.format(epoch+1, setting.max_epoch, iteration + 1, epoch_iters, running_loss / setting.display_interval, np.mean(batch_time), lr))
                batch_time = []
                running_loss = 0.0

        # Validation
        if (epoch+1) % setting.val_interval == (setting.val_interval - 1):
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
    
    rootPath = "D:/Dataset/Classification/cifar10/"
    imgPath = rootPath + "_Images/"
    train_fListPath = rootPath + "train.txt"
    val_fListPath = rootPath + "val.txt"

    model_name = "ResNet50"
    save_folder = "E:/Coding/pytorch/project/Classification_Pytorch/weights/ResNet50_origWrong/aug_LRdecay_bs128_ep200_warm_lr0.05_gamma0.1/"
    best_model_path = os.path.join(save_folder, model_name + "_Best.pth")

    num_classes = 10
    batch_size_train = 128
    batch_size_val = 100
    max_epoch = 200
    display_interval = 100
    val_interval = 10

    base_lr = 0.05 # 0.01
    gamma = 0.1
    lr_decay_steps = [80, 120] # [80]
    warm_epoch = 5

    #--------------------------------------------------------------------------------------------------------#

    setting = Setting(num_classes, batch_size_train, batch_size_val, max_epoch, display_interval, val_interval, 
                      base_lr, gamma, lr_decay_steps, warm_epoch)

    # Setup output information 
    os.makedirs(save_folder, exist_ok=True)
    log_file_path = os.path.join(save_folder, model_name + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log')
    
    outputManage = OutputManager(log_file_path)

    outputManage.output("\nHyper-parameter setting:")
    lr_decay_steps_str = "["
    for s in lr_decay_steps: lr_decay_steps_str += (str(s) + ",")
    lr_decay_steps_str = lr_decay_steps_str[:-1] + "]"

    outputManage.output(" - lr = {}\n - gamma = {}\n - lr_decay_steps = {}\n - batch_size = {}\n - max_epoch = {}\n".format(
        base_lr, gamma, lr_decay_steps_str, batch_size_train, max_epoch))

    # Loading and normalizing Custom cifar10 dataset
    outputManage.output("Setup dataset ...")
    transform = transforms.Compose(
        [transforms.Resize(size=(60,60)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomCrop(size=(48,48)),
         transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])

    transform_val = transforms.Compose(
        [transforms.Resize(size=(60,60)),
         transforms.CenterCrop(size=(54,54)),
         transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])

    train_Dataset = cDataset.ClassifyDataset(train_fListPath, imgPath, transform=transform)
    train_Loader = DataLoader(train_Dataset, batch_size=batch_size_train, shuffle=True, num_workers=0)
    val_Dataset = cDataset.ClassifyDataset(val_fListPath, imgPath, transform=transform_val)
    val_Loader = DataLoader(val_Dataset, batch_size=batch_size_val, shuffle=False, num_workers=0)

    num_train = len(train_Dataset)
    epoch_iters = num_train // batch_size_train
    
    # Setup model
    outputManage.output("Create Model: {}".format(model_name))
    if model_name=="VGG16":
        net = VGG.VGG16(setting.num_classes, init_weights=True)
    elif model_name=="VGG19":
        net = VGG.VGG19(setting.num_classes, init_weights=True)
    elif model_name=="ResNet50":
        net = ResNet.ResNet50(setting.num_classes, init_weights=True)
    elif model_name=="ResNet101":
        net =  ResNet.ResNet101(setting.num_classes, init_weights=True)
    elif model_name=="ResNet50_official":
        import torchvision.models as models
        net = models.resnet50()
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
    
    train(net, train_Loader, val_Loader, device, setting, epoch_iters, outputManage, best_model_path)

#--------------------------------------------------------------------------------------------------------#  
    
if __name__ == '__main__':
    main()