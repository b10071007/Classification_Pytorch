import os

import torch
from torch.utils.data import Dataset
from skimage import io

# ------------------------------------------------------------------------------#

def seperateMapList(mapList):
    """
    Separate mapping list into file list and label list.

    :param mapList: mapping list being read
    :return: fList: file list with relative path
    :return: labels: label list
    """
    fList = []
    labels = []
    for line in mapList:
        if line[-1]=='\n': line = line[:-1]
        file, label = line.split(' ')
        fList.append(file)
        labels.append(label)
    return fList, labels

# ------------------------------------------------------------------------------#

class ClassifyDataset(Dataset):
    """ Classification dataset """

    def __init__(self, fListDir, imgPath, transform=None):
        """
        Args:
            fListDir (string): Directory to the file list with annotations.
            imgPath (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.fListDir = fListDir
        self.imgPath = imgPath
        self.transform = transform

        f_obj = open(fListDir, 'r')
        mapList = f_obj.readlines()
        f_obj.close()

        fList, labels = seperateMapList(mapList)
        self.fList = fList
        self.labels = labels

    def __len__(self):
        return len(self.fList)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.imgPath, self.fList[idx])
        image = io.imread(img_name)
        label = torch.tensor(int(self.labels[idx]))

        if self.transform:
            image = self.transform(image)

        return image, label
