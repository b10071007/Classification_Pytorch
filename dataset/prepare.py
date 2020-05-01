# -*- coding: utf-8 -*-
"""
Functions for prepare dataset
"""
import os
import numpy as np
# from random import shuffle
import cv2
from PIL import Image
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing
from dataset import augment

# ---------------------------------------------------------------------------------------------------------------------#

''' Split dataset into train/val/test '''

def shuffle_Flist(img_list):
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)
    np.random.shuffle(img_list)


def get_class_and_counts(fList):
    # get class for each sample
    class_list = []
    for l in range(len(fList)):
        line = fList[l]
        class_l = line.split(' ')[1]
        if '\n' in class_l:
            class_l = class_l.split('\n')[0]
        class_list.append(class_l)

    class_list = np.array(class_list)
    per_class, counts = np.unique(class_list, return_counts=True)
    return class_list, per_class, counts

def Split_set_by_class_distribution(srcFPath, setNameList=None, ratioList=None, numList=None):
    print('\nSplit partition by class distribution...')
    rootPath = os.path.split(srcFPath)[0]
    with open(srcFPath, 'r') as fObj:
        srcFList = fObj.readlines()

    srcFList = np.array(srcFList)
    num_file = len(srcFList)

    isExist_numList = numList is not None
    isExist_ratioList = ratioList is not None

    if (numList is None) and (ratioList is None):  # No setting
        raise ValueError('Not set ratio and number at the same time!')
    elif numList is not None:  # Set number
        if np.sum(numList) > num_file:
            raise ValueError('The sum of numList is larger than data size: {}.'.format(num_file))
        else:
            ratioList = [num / num_file for num in numList]
    elif ratioList is not None:  # Set ratio
        if np.sum(ratioList) != 1:
            raise ValueError('The sum of proportion is not equal to 1.')

    shuffle_Flist(srcFList)
    class_list, per_class, counts = get_class_and_counts(srcFList)

    num_set = len(setNameList)
    fList_set = [[] for i in range(num_set)]

    cumulative = np.append([0], np.cumsum(ratioList))

    for c in per_class:
        FList_c = srcFList[class_list == c]
        num_total = len(FList_c)
        end_point = [int(cumulative[i] * num_total) for i in range(num_set + 1)]

        for idx_set in range(num_set):
            fList_set[idx_set].extend(FList_c[end_point[idx_set]:end_point[idx_set + 1]])

    # Shuffle the input file list.
    for idx_set in range(num_set):
        shuffle_Flist(fList_set[idx_set])

    # Copy the top # of files as test files.
    for i in range(num_set):
        flist = fList_set[i]
        size = len(flist)
        print("The number of samples for \"{}\": {}".format(setNameList[i], size))
        # print('For "{}"'.format(setNameList[i]))
        # print('The number of samples is {}'.format(size))
        if size == 0:
            print('The file list is empty, so it is not outputted.')
            continue
        destFObj = open(os.path.join(rootPath, setNameList[i]), "w")
        for line in flist:
            destFObj.write(line)
        destFObj.close()


def Split_train_test_by_dir(rootPath, srcFileList, keywords, output_dir):
    print('\nSplit training and testing set by directories...')
    srcPath = rootPath + srcFileList
    srcFObj = open(srcPath, "r")
    srcFList = srcFObj.readlines()
    srcFObj.close()
    srcFList = np.array(srcFList)

    trainFlist = []
    testFlist = []
    invalid = []
    for line in srcFList:
        if keywords[0] in line:
            trainFlist.append(line)
        elif keywords[1] in line:
            testFlist.append(line)
        else:
            invalid.append(line)

    Split_list = [trainFlist, testFlist, invalid]
    # Copy the top # of files as test files.
    output_dir.append('invalid.txt')
    for i in range(3):
        destFObj = open(rootPath + output_dir[i], "w")
        for line in Split_list[i]:
            destFObj.write(line)
        destFObj.close()
    print('Done.')

# -----------------------------------------------------------------------------#

''' Read images '''

def _read_each(img_file):
    img = cv2.imread(img_file)  # shape: [H, W, BGR]
    return img


def one_hot_encode(labels):
    num_label = len(np.unique(labels))
    one_hot_label = (np.arange(num_label) == labels[:, None].astype(int)).astype(int)
    return one_hot_label


def Read_images(img_list):
    checkFile = img_list[0]
    fileSplit = checkFile.split(' ')
    if len(fileSplit) == 2:  # map.txt with label
        with_label = True
        # labels = img_list[:, 1]
        # img_list = img_list[:, 0]
        labels = [file.split(' ')[1] for file in img_list]
        img_list = [file.split(' ')[0] for file in img_list]
    elif len(img_list.shape) == 1:  # map.txt without label
        with_label = False
        img_list = [file.split(' ')[0] for file in img_list]
    else:
        raise ValueError('The number of columns in map.txt is wrong.')

    num_cpus = multiprocessing.cpu_count()
    pool = ThreadPool(num_cpus)
    images = pool.map(_read_each, img_list)
    pool.close()
    pool.join()

    if with_label:
        one_hot_label = one_hot_encode(np.array(labels))
    else:
        one_hot_label = []
    return [images, one_hot_label]


# -----------------------------------------------------------------------------#

''' Compute mean '''

def Compute_mean(mapListPath, srcImagePath, new_size, channel, outFilePath, outMeanImagePath):
    print("\nSource mapping file = %s" % mapListPath)
    with open(mapListPath, 'r') as fObj:
        img_list = fObj.readlines()
    img_list = [srcImagePath + file for file in img_list]
    data_size = len(img_list)
    batch_size = 200
    iteration = data_size / batch_size

    if iteration == int(iteration):
        batch_list = np.repeat(batch_size, iteration)
        iteration = int(iteration)
    else:
        batch_list = np.repeat(batch_size, int(iteration) + 1)
        batch_list[-1] = batch_size - ((int(iteration) + 1) * batch_size - data_size)
        iteration = int(iteration) + 1
    meanImage = np.zeros((new_size[0], new_size[1], channel), dtype=np.float64)
    squaredImage = np.zeros((new_size[0], new_size[1], channel), dtype=np.float64)

    Aug = augment.Augmenter(new_size, None, None, None, None, None, None, None)

    print('\rComputing image mean... ({} / {})'.format(0, data_size), end='')
    for it in range(iteration):
        idx = [it * batch_size, (it + 1) * batch_size]
        img_list_batch = img_list[idx[0]:idx[1]]
        img_batch, _ = Read_images(img_list_batch)

        img_aug = Aug.Augment(img_batch)
        img_aug = np.array(img_aug)

        meanImage += img_aug.sum(axis=0)
        squaredImage += (img_aug.astype(float) * img_aug.astype(float)).sum(axis=0)
        print('\rComputing image mean... ({} / {})'.format((it + 1) * batch_size, data_size), end='')
    print('\rComputing image mean... ({} / {})'.format(data_size, data_size))
    meanImage = meanImage / data_size
    squaredImage = squaredImage / data_size

    # Channel means
    channel_mean = meanImage.mean(axis=(0, 1))
    channel_mean = channel_mean.round(4)
    averaged_channel_mean = channel_mean.mean()
    averaged_channel_mean = averaged_channel_mean.round(4)

    print('Done.\n')
    print('#----------------------------------------------#')
    print("Channel means = %0.4f, %0.4f, %0.4f" %
          (channel_mean[0], channel_mean[1], channel_mean[2]))
    print("Averaged Channel Mean = %0.4f" % averaged_channel_mean)

    # Channel std values
    stdImage = squaredImage - meanImage * meanImage
    channel_std = np.sqrt(stdImage.mean(axis=(0, 1)))
    channel_std = channel_std.round(4)
    averaged_channel_std = channel_std.mean()
    averaged_channel_std = averaged_channel_std.round(4)

    print("Channel STDs  = %0.4f, %0.4f, %0.4f" %
          (channel_std[0], channel_std[1], channel_std[2]))
    print("Averaged Channel STD = %0.4f" % averaged_channel_std)
    print('#----------------------------------------------#')
    # Dump to files : channel means and std values.
    outFObj = open(outFilePath, "w")
    strMsg = "Channel means (in B/G/R order) : \n" + \
             ("[%0.4f, %0.4f, %0.4f]" % (channel_mean[0], channel_mean[1], channel_mean[2])) + "\n" + \
             ("Averaged Channel Mean = %0.4f" % (averaged_channel_mean))
    outFObj.write(strMsg)
    outFObj.write("\n\n")
    strMsg = "Channel STD values (in B/G/R order) : \n" + \
             ("[%0.4f, %0.4f, %0.4f]" % (channel_std[0], channel_std[1], channel_std[2])) + "\n" + \
             ("Averaged Channel STD = %0.4f" % (averaged_channel_std))
    outFObj.write(strMsg)
    outFObj.write("\n")
    outFObj.close()

    # Convert to uint8 and then dump the averaged image to the file.
    aveImg = Image.fromarray(meanImage.astype(np.uint8))
    aveImg.save(outMeanImagePath)

# -----------------------------------------------------------------------------#

''' Batch setting '''

def Batch_setting(img_list, batch_size):
    data_size = len(img_list)
    iteration = data_size / batch_size
    if iteration == int(iteration):
        batch_list = np.repeat(batch_size, iteration)
        iteration = int(iteration)
    else:
        batch_list = np.repeat(batch_size, int(iteration) + 1)
        batch_list[-1] = batch_size - ((int(iteration) + 1) * batch_size - data_size)
        iteration = int(iteration) + 1
    return batch_list, iteration
