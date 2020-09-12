import os
import time
import numpy as np
import torch
# import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
sys.path.append("E:/Coding/pytorch/project/Classification_Pytorch/")
from dataset import classifyDataset as cDataset
import utils
import models


def test(net, test_Loader, device, display_interval, save_folder, num_classes, iterations):
    
    print('{}: Start testing  '.format(utils.GetCurrentTime()))
    confusionMatrix = np.zeros((num_classes,num_classes))
    proba_all = []
    label_all = []
    # predict_proba = []
    predict_class = []
    
    start_time = time.time()
    with torch.no_grad():
        for iter_val, data in enumerate(test_Loader):
            if iter_val % display_interval == (display_interval-1):
                print("Iterations: [{}/{}]".format(iter_val+1, iterations))

            # images, labels = data
            images, labels = data[0].to(device), data[1]
            outputs = net(images)
            proba = utils.softmax(outputs.cpu())
            predicted = torch.max(proba, 1)
            
            proba_all.extend(proba.numpy())
            label_all.extend(labels.numpy())
            # predict_proba.extend(predicted[0].numpy())
            predict_class.extend(predicted[1].numpy())
            
            for idx, label in enumerate(labels):
                confusionMatrix[label,predicted[1][idx]] += 1 

        print("Iterations: [{}/{}]".format(iterations, iterations))
    
    end_time = time.time()
    
    print('{}: Testing Finished'.format(utils.GetCurrentTime()))
    
    data_size = len(predict_class)
    
    proba_all = np.array(proba_all)
    label_all = np.reshape(label_all, (-1,1))
    predict_class = np.reshape(predict_class, (-1,1))
    
    accuracy = np.sum(np.diag(confusionMatrix)) / data_size
    error = np.round(1-accuracy, 6)
    accuracy_perClass = np.diag(confusionMatrix) / np.sum(confusionMatrix, axis=0)
    error_perClass = 1-accuracy_perClass
    
    output = np.concatenate([proba_all, label_all, predict_class], axis=1)
    output = output.astype('float')

    probaMatrixPath = os.path.join(save_folder, "TestProbaMatrix.txt")
    confMatPath = os.path.join(save_folder, '{}_Err={}.txt'.format("TestConfMat", error))
    
    np.savetxt(probaMatrixPath, output, fmt='%f')  
    np.savetxt(confMatPath, confusionMatrix, fmt='%d', delimiter='\t') 
    
    time_per_img = (end_time - start_time)*1000/data_size
    
    print("\n#------------------------------------------------------#\n")
    print("Overall Accuracy : {}%".format(round(accuracy*100,2)))
    print("Overall Error Rate : {}%".format(round(error*100,2)))
    print('Test time per image (ms): {}'.format(round(time_per_img,4)))
    print()
    print("Dump probability matrix: {}".format(os.path.split(probaMatrixPath)[-1]))
    print("Dump confusion matrx: {}".format(os.path.split(confMatPath)[-1]))

    print('\n=============================')
    print('Error Rate (per-class): ')
    for l in range(num_classes):
        print('  class {} : {}%'.format(l, round(error_perClass[l]*100,2)))
    print('=============================\n')

def main():
        
    rootPath = "D:/Dataset/Classification/cifar10/"
    imgPath = rootPath + "_Images/"
    test_fListPath = rootPath + "test.txt"

    model_name = "WRN_N4_k4"
    save_folder = "./weights/allTrain/WRN_N4_k4_drop0.5/bs128_ep200_warm5_lr0.1_gamma0.2_wdecay0.0005_nesterov/"
    model_path = os.path.join(save_folder, model_name + "_Best.pth")

    num_classes = 10
    batch_size_test = 100
    display_interval = 40 

    #--------------------------------------------------------------------------------------------------------#

    ''' Loading and normalizing Custom cifar10 dataset '''
    transform_test = transforms.Compose(
        [
         transforms.Resize(size=(60,60)),
         transforms.CenterCrop(size=(54,54)),
         transforms.ToTensor(),
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
         ])

    test_Dataset = cDataset.ClassifyDataset(test_fListPath, imgPath, transform=transform_test)
    test_Loader = DataLoader(test_Dataset, batch_size=batch_size_test, shuffle=False, num_workers=2)
    num_data = len(test_Dataset)
    iterations = np.ceil(num_data/batch_size_test).astype(int)

    ''' Setup model '''
    model_names = sorted( name[6:] for name in models.__dict__
                          if name.startswith("Build_")
                          and callable(models.__dict__[name]) )
    
    # print("Support models:")
    # for m in model_names: 
    #     print(' - ' + m)
        
    print("Create Model: {}".format(model_name))

    net = None
    for model_names_each in model_names:
        if model_name == model_names_each:
            net = models.__dict__["Build_" + model_names_each](
                        num_classes = num_classes,
                        init_weights = False)
    if net is None:
        raise ValueError("Not support model -> \"{}\"".format(model_name))

    state_dict = torch.load(model_path)
    net.load_state_dict(state_dict)
    net.eval()

    print(' - Finished loading model!')

    ''' Setup GPU '''
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(" - GPU is available -> use GPU")
    else:
        device = torch.device("cpu")
        print(" - GPU is not available -> use GPU")

    net.to(device)

    test(net, test_Loader, device, display_interval, save_folder,num_classes, iterations)

#--------------------------------------------------------------------------------------------------------#  
    
if __name__ == '__main__':
    main()