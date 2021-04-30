
import os
import numpy as np
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import sys
sys.path.append("E:/Coding/pytorch/project/Classification_Pytorch/")
from dataset import classifyDataset as cDataset
from utils import OutputManager, GetCurrentTime, prepareAugmentation
import models

#--------------------------------------------------------------------------------------------------------#

# Setup settings and hyper-parameter
# class Setting():
#     def __init__(self, num_classes, batch_size_train, batch_size_val, max_epoch, display_interval, val_interval,
#                  base_lr, gamma, lr_decay_steps, warm_epoch, weight_decay, nesterov):
#         self.num_classes = num_classes
#         self.batch_size_train = batch_size_train
#         self.batch_size_val = batch_size_val
#         self.max_epoch = max_epoch
#         self.display_interval = display_interval
#         self.val_interval = val_interval
#         self.base_lr = base_lr
#         self.gamma = gamma
#         self.lr_decay_steps = lr_decay_steps
#         self.warm_epoch = warm_epoch
#         self.weight_decay = weight_decay
#         self.nesterov = nesterov

#--------------------------------------------------------------------------------------------------------#
        
def ParseTrainArgs():

    parser = argparse.ArgumentParser(description='Classification Training')

    # Dataset setting
    parser.add_argument('-img_dir', help='the directory for images' )
    parser.add_argument('-fList_train', help='the mappling list for training set' )
    parser.add_argument('-fList_val', help='the mappling list for validation set' )
    parser.add_argument('-num_classes', type=int, default=10, help='the number of classes' )
    parser.add_argument('-out_dir', default='./weights/', help='the output directory (training logs and trained model)')
    parser.add_argument('-num_workers', type=int, default=8,  help='how many subprocesses to use for data loading.')
    
    # Training setting
    parser.add_argument('-gpu_id', default='0', help='setup visible gpus, for example 0,1')
    parser.add_argument('-model_name', help='the classification model')
    parser.add_argument('-pretrained', default='None', help='the path of pre-trained model')
    parser.add_argument('-stride_times', type=int, default=5, help='flexible architechtures according to different strides times (e.g. stride_times=5 means reduce 2^5=32)')
    parser.add_argument('-batch_size_train', type=int, default=64, help='the batch size for training')
    parser.add_argument('-batch_size_val', type=int, default=100, help='the batch size for validation')
    parser.add_argument('-epochs', type=int, default=300, help='the total training epochs')
    parser.add_argument('-val_epochs', type=int, default=10, help='the frequency to validate model')
    parser.add_argument('-display_interval', type=int, default=100,  help='the total training epochs')
    
    # Data augmentation and normalization setting
    parser.add_argument("-resize", type=int, nargs='+', default=None, help='resize training images (W,H)')
    parser.add_argument("-random_resize", action='store_true', default=False, 
                        help='randomly resize training image instead of fixed size')
    parser.add_argument("-random_resize_scale", type=float, nargs='+', default=(0.08, 1.0), 
                        help='scale range of size of the origin size cropped')
    parser.add_argument("-random_resize_ratio", type=float, nargs='+', default=(0.75, 1.33), 
                        help='resize training images with delta (W,H)')
    parser.add_argument("-random_flip_H", action='store_true', default=False, help='horizontally flip training images')
    parser.add_argument("-random_flip_V", action='store_true', default=False, help='vertically flip training images')
    parser.add_argument("-random_crop_size", type=int, nargs='+', default=(32, 32), help='crop training images')
    parser.add_argument("-random_rotation", type=int, nargs='+', default=None, help='random rotation with self-defined angles, e.q. (0, 90, 180, 270)')
    parser.add_argument("-color_jitter_factor", type=float, nargs='+', default=(0, 0, 0, 0), help='color jittering for training images [brightness, contrast, saturation, hue]')

    parser.add_argument("-resize_val", type=int, nargs='+', default=None, help='resize validation images (W,H)')
    parser.add_argument("-center_crop_val", type=int, nargs='+', default=None, help='center crop validation images (W,H)')

    parser.add_argument('-channel_mean', type=float, nargs='+', default=(0.4914, 0.4822, 0.4465), 
                        help='the channel mean of images for training set')
    parser.add_argument('-channel_std', type=float, nargs='+', default=(0.2023, 0.1994, 0.2010), 
                        help='the channel standard deviation of images for training set')

    # Optimization setting
    parser.add_argument('-base_lr', type=float, default=0.1, help='the learning rate')
    parser.add_argument('-gamma', type=float, default=0.1, help='the coefficient for learning rate step decay')
    parser.add_argument('-lr_decay_steps', type=int, nargs='+', default=[150, 225], help='the steps for step decay')
    parser.add_argument('-weight_decay', type=float, default=0.0005, help='the weight decay coefficient')
    parser.add_argument('-nesterov', action='store_true', default=False, help='use nesterov momentum')
    parser.add_argument('-warm_epoch', type=int, default=1, help='the epochs to warm up')

    args = parser.parse_args()
    return args

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
    outputManage.output( 'Accuracy on validation images: {} %'.format(round(accuracy,2)))
    
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


def train(net, train_Loader, val_Loader, device, args, epoch_iters, outputManage, best_model_path):
    
    batch_time = []
    log_str = "Epoch: [{:3d}/{:3d}] Iterations: [{:3d}/{:3d}] Loss: {:.3f} Batch_time: {:.2f} ms LR: {:.4f}"

    start_training = time.time()

    best_ep = 0
    best_acc = 0
    best_model_message = "Best model until now:\n - epoch={}\n - Accuracy={}"

    # Setup optimizer
    step_index = 0
    lr = args.base_lr
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=args.nesterov, weight_decay=args.weight_decay)

    outputManage.output('\n{}: Start Training '.format(GetCurrentTime()))
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        
        if (epoch+1) in args.lr_decay_steps:
            step_index += 1

        net.train()
        running_loss = 0.0
        for iteration, data in enumerate(train_Loader):
            
            lr = adjust_learning_rate(optimizer, args.base_lr, args.gamma, step_index, 
                                      epoch, args.warm_epoch, iteration, epoch_iters)

            torch.cuda.synchronize()
            start = time.time()*1000

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            end = time.time()*1000

            batch_time.append(end-start)

            # print statistics
            running_loss += loss.item()
            if iteration % args.display_interval == (args.display_interval - 1):  # print every 'display_interval' mini-batches
                outputManage.output(
                    log_str.format(epoch+1, args.epochs, iteration + 1, epoch_iters, running_loss / args.display_interval, np.mean(batch_time), lr))
                batch_time = []
                running_loss = 0.0

        # Validation
        net.eval()
        if epoch % args.val_epochs == (args.val_epochs - 1):
            accuracy = val(net, val_Loader, device, outputManage)
            if accuracy > best_acc:
                best_acc = accuracy
                best_ep = epoch
                torch.save(net.state_dict(), best_model_path)

            outputManage.output(best_model_message.format(best_ep, best_acc) + '\n')


    end_training = time.time()
    trainingTime = round((end_training - start_training)/60)
    outputManage.output('{}: Training Finished  '.format(GetCurrentTime()))
    outputManage.output('Training time: {} mins'.format(trainingTime))
    
    best_acc = round(best_acc, 4)
    with open(os.path.split(best_model_path)[0] + '/Best_model_{}_{}.txt'.format(best_ep, best_acc), 'w') as fObj:
        fObj.writelines("{}\t{}".format(best_ep, best_acc))

#--------------------------------------------------------------------------------------------------------#

def main():
    
    args = ParseTrainArgs()

    ''' Setup output information '''
    os.makedirs(args.out_dir, exist_ok=True)
    best_model_path = os.path.join(args.out_dir, args.model_name + "_Best.pth")
    log_file_path = os.path.join(args.out_dir, args.model_name + time.strftime('_%Y-%m-%d-%H-%M', time.localtime(time.time())) + '.log')
    outputManage = OutputManager(log_file_path)

    outputManage.output("\nHyper-parameter setting:")
    lr_decay_steps_str = "["
    for s in args.lr_decay_steps: lr_decay_steps_str += (str(s) + ",")
    lr_decay_steps_str = lr_decay_steps_str[:-1] + "]"

    outputManage.output(" - lr = {}\n - gamma = {}\n - lr_decay_steps = {}\n - batch_size = {}\n - epochs = {}\n".format(
        args.base_lr, args.gamma, lr_decay_steps_str, args.batch_size_train, args.epochs))
        
    ''' Loading and normalizing Custom dataset '''
    outputManage.output("Setup dataset ...")
    transform = transforms.Compose(prepareAugmentation(args, is_train=True))
    transform_val = transforms.Compose(prepareAugmentation(args, is_train=False))

    train_Dataset = cDataset.ClassifyDataset(args.fList_train, args.img_dir, transform=transform)
    train_Loader = DataLoader(train_Dataset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    val_Dataset = cDataset.ClassifyDataset(args.fList_val, args.img_dir, transform=transform_val)
    val_Loader = DataLoader(val_Dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=args.num_workers)

    num_train = len(train_Dataset)
    epoch_iters = num_train // args.batch_size_train

    ''' Setup model '''
    outputManage.output("Create Model: {}".format(args.model_name))

    model_names = sorted( name[6:] for name in models.__dict__
                          if name.startswith("Build_")
                          and callable(models.__dict__[name]) )
    
    # print("Support models:")
    # for m in model_names: 
    #     print(' - ' + m)

    net = None
    for model_names_each in model_names:
        if args.model_name == model_names_each:
            net = models.__dict__["Build_" + model_names_each](num_classes = args.num_classes, stride_times=args.stride_times, init_weights = True)
            break
    if net is None:
        raise ValueError("Not support model -> \"{}\"".format(args.model_name))

    ''' Load pre-trained model'''
    if args.pretrained != "None":
        # Get state_dict of built model and pre-trained model
        state_dict_built = net.state_dict()
        state_dict_pretrain = torch.load(args.pretrained)
        
        # Preprocess layer names and get the number of classes of pre-trained model
        from collections import OrderedDict
        new_state_dict_pretrain = OrderedDict()
        num_classes_pretrain = args.num_classes
        
        for k, v in state_dict_pretrain.items():
            if k[:7] == 'module.':
                name = k[7:]  
            else:
                name = k
            
            # if "num_classes_pretrain" is different from "num_classes" -> discard fc layer
            if "fc" in name:
                num_classes_pretrain = v.shape[0]
                if num_classes_pretrain != args.num_classes:
                    continue
            
            new_state_dict_pretrain[name] = v
                
        state_dict_built.update(new_state_dict_pretrain)
        net.load_state_dict(state_dict_built)
        print("Load pre-trained model:", args.pretrained)
        
    else:
        print("Training from scratch")
        
    
    ''' Setup GPU '''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if torch.cuda.is_available():
        device = torch.device("cuda")
        outputManage.output(" - GPU is available -> use GPU")
        net.to(device)
        outputManage.output(" - Used GPU: " + args.gpu_id)
        net = nn.DataParallel(net) # multi-GPU
    else:
        device = torch.device("cpu")
        outputManage.output(" - GPU is not available -> use GPU")
        net.to(device)

    train(net, train_Loader, val_Loader, device, args, epoch_iters, outputManage, best_model_path)

#--------------------------------------------------------------------------------------------------------#  
    
if __name__ == '__main__':
    main()