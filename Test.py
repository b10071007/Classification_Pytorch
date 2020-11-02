import os
import time
import numpy as np
import argparse
from collections import OrderedDict
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
sys.path.append("E:/Coding/pytorch/project/Classification_Pytorch/")
from dataset import classifyDataset as cDataset
import utils
import models


def test(net, test_Loader, device, display_interval, out_dir, num_classes, iterations):
    
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

    probaMatrixPath = os.path.join(out_dir, "TestProbaMatrix.txt")
    confMatPath = os.path.join(out_dir, '{}_Err={}.txt'.format("TestConfMat", error))
    
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
    
    parser = argparse.ArgumentParser(description='Classification Testing')

    # Dataset setting
    parser.add_argument('-img_dir', help='the directory for images' )
    parser.add_argument('-fList_test', help='the mappling list for training set' )
    parser.add_argument('-num_classes', type=int, default=10, help='the number of classes' )
    parser.add_argument('-out_dir', default='./weights/', help='the output directory (training logs and trained model)')
    parser.add_argument('-model_path', default='None', help='the output directory (training logs and trained model)')
    
    parser.add_argument('-channel_mean', type=float, nargs='+', default=(0.4914, 0.4822, 0.4465), 
                        help='the channel mean of images for training set')
    parser.add_argument('-channel_std', type=float, nargs='+', default=(0.2023, 0.1994, 0.2010), 
                        help='the channel standard deviation of images for training set')

    # Testing setting
    parser.add_argument('-gpu_id', default='0', help='setup visible gpus, for example 0,1')
    parser.add_argument('-model_name', help='the classification model')
    parser.add_argument('-batch_size_test', type=int, default=64, help='the batch size for testing')
    parser.add_argument('-display_interval', type=int, default=100,  help='the total training epochs')

    args = parser.parse_args()
    
    # rootPath = "D:/Dataset/Classification/cifar10/"
    # imgPath = rootPath + "_Images/"
    # test_fListPath = rootPath + "test.txt"

    # model_name = "WRN_N4_k4"
    # save_folder = "./weights/allTrain/WRN_N4_k4_drop0.3/bs128_ep200_warm5_lr0.1_gamma0.2_wdecay0.0005_nesterov/"
    # model_path = os.path.join(save_folder, model_name + "_Best.pth")

    # num_classes = 10
    # batch_size_test = 100 # data size can be not divisible by batch_size
    # display_interval = 40 

    #--------------------------------------------------------------------------------------------------------#

    ''' Loading and normalizing Custom dataset '''
    transform_test = transforms.Compose(
        [
        #  transforms.Resize(size=(60,60)),
        #  transforms.CenterCrop(size=(54,54)),
         transforms.ToTensor(),
         transforms.Normalize(args.channel_mean, args.channel_std),
         ])

    test_Dataset = cDataset.ClassifyDataset(args.fList_test, args.img_dir, transform=transform_test)
    test_Loader = DataLoader(test_Dataset, batch_size=args.batch_size_test, shuffle=False, num_workers=2)
    num_data = len(test_Dataset)
    iterations = np.ceil(num_data/args.batch_size_test).astype(int)

    ''' Setup model '''
    model_names = sorted( name[6:] for name in models.__dict__
                          if name.startswith("Build_")
                          and callable(models.__dict__[name]) )
    
    # print("Support models:")
    # for m in model_names: 
    #     print(' - ' + m)
        
    print("Create Model: {}".format(args.model_name))

    net = None
    for model_names_each in model_names:
        if args.model_name == model_names_each:
            net = models.__dict__["Build_" + model_names_each](
                        num_classes = args.num_classes,
                        init_weights = False)
    if net is None:
        raise ValueError("Not support model -> \"{}\"".format(args.model_name))

    if args.model_path == 'None':
        model_path = os.path.join(args.out_dir, args.model_name + "_Best.pth")
    else:
        model_path = args.model_path

    # Load model weight
    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.eval()

    print(' - Finished loading model!')

    ''' Setup GPU '''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(" - GPU is available -> use GPU")
    else:
        device = torch.device("cpu")
        print(" - GPU is not available -> use GPU")

    net.to(device)

    test(net, test_Loader, device, args.display_interval, args.out_dir, args.num_classes, iterations)

#--------------------------------------------------------------------------------------------------------#  
    
if __name__ == '__main__':
    main()