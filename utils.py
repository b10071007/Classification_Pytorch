import os
import numpy as np
from datetime import datetime

import torchvision.transforms as transforms

#----------------------------------------------------------------------------------------------------------#

def GetCurrentTime():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

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

def softmax(x):
    e_x = np.exp(x)
    return e_x / e_x.sum(1).reshape(e_x.shape[0],1)

#----------------------------------------------------------------------------------------------------------#

##############################
#----- Data augmetation -----#
##############################

class RandomRotationSetAngles:
    """Rotate by one of the given angles."""

    def __init__(self, angles=(0, 90, 180, 270)):
        self.angles = angles

    def __call__(self, x):
        angle = np.random.choice(self.angles)
        return transforms.functional.rotate(x, angle)

def prepareAugmentation(args, is_train=True):

    augmentationList = []
    
    if (is_train):
        #----- Augmentation for training -----#
        # Fixed Resize
        if (args.resize is not None and args.random_resize != True):
            augmentationList.extend(   
                [
                    transforms.Resize(size=args.resize),
                    transforms.RandomCrop(size=args.random_crop_size),
                ]
            )
        # Random Resize
        elif(args.random_resize == True):
            resizeFirst = np.array(args.random_crop_size) * args.random_resize_scale[1] * args.random_resize_ratio[1]
            resizeFirst = resizeFirst.astype(int)
            augmentationList.extend(   
                [
                    transforms.Resize(size=resizeFirst),
                    transforms.RandomResizedCrop(size=args.random_crop_size, scale=args.random_resize_scale, ratio=args.random_resize_ratio),
                ]
            )
        # Flipping
        if (args.random_flip_H == True):
            augmentationList.append(transforms.RandomHorizontalFlip())
        if (args.random_flip_V == True):
            augmentationList.append(transforms.RandomVerticalFlip())

        # Rotation
        if  args.random_rotation is not None:
            augmentationList.append(RandomRotationSetAngles(args.random_rotation))

        # Color jittering
        if (args.color_jitter_factor is not None):
            augmentationList.append(
                transforms.RandomApply(
                    [transforms.ColorJitter(args.color_jitter_factor[0], args.color_jitter_factor[1], args.color_jitter_factor[2], args.color_jitter_factor[3])],
                    p=0.5)
            )
    else:
        #----- Augmentation for validation -----#
        # Fixed Resize
        if (args.resize_val is not None):
            augmentationList.append( transforms.Resize(size=args.resize_val) )
        if (args.center_crop_val is not None):
            augmentationList.append( transforms.CenterCrop(size=args.center_crop_val) )
    
    # Normalize
    augmentationList.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(args.channel_mean, args.channel_std),
        ]
    )

    return augmentationList
