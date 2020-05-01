# -*- coding: utf-8 -*-]
"""
Functions for data augmentation
"""
import numpy as np
import cv2
from multiprocessing.dummy import Pool as ThreadPool
import multiprocessing

# ---------------------------------------------------------------------------------------------------------------------#

''' Data Augmentation '''


# augment function
def Resize(img, new_size, delta_size):
    if new_size is not None:
        if delta_size is not None:
            delta_h = delta_size[0]
            delta_w = delta_size[1]
        else:
            delta_h = 0
            delta_w = 0
        set_h = new_size[0] + np.random.randint(-delta_h, delta_h + 1)
        set_w = new_size[1] + np.random.randint(-delta_w, delta_w + 1)
        img = cv2.resize(img, (set_h, set_w), interpolation=cv2.INTER_CUBIC)

    return img


def Crop(img, crop_size, crop_type='random'):
    if crop_size is not None:
        img_size = list(img.shape[:2])

        if crop_type == 'random':
            offset = np.array(img_size) - np.array(crop_size)
            start_h = 0
            start_w = 0
            if offset[0] != 0:
                start_h = int(np.random.randint(0, offset[0], 1))
            if offset[1] != 0:
                start_w = int(np.random.randint(0, offset[1], 1))
            crop_img = img[start_h:(start_h + crop_size[0]), start_w:(start_w + crop_size[1])]

        elif crop_type == 'center':
            offset = ((np.array(img_size) - np.array(crop_size)) / 2).astype(int)
            crop_img = img[offset[0]:(offset[0] + crop_size[0]), offset[1]:(offset[1] + crop_size[1]), :]

        else:
            raise ValueError('The crop_type setting "{}" is unknown'.format(crop_type))
    else:
        crop_img = img

    return crop_img


# def Random_Crop(img, crop_size):
#    if crop_size!=None:
#        img_size = list(img.shape[:2])
#        offset = np.array(img_size) - np.array(crop_size)
#        start_h = 0
#        start_w = 0
#        if offset[0]!=0:
#            start_h = int(np.random.randint(0, offset[0], 1))
#        if offset[1]!=0:
#            start_w = int(np.random.randint(0, offset[1], 1))
#
#        crop_img = img[start_h:(start_h+crop_size[0]), start_w:(start_w+crop_size[1])]
#    else:
#        crop_img = img
#
#    return crop_img

def Rotate(img, rotate=360):
    if rotate is None:
        img_rotated = img
    else:
        (h, w) = img.shape[:2]
        if rotate == 360:
            angle_list = [0, 90, 180, 270]
            set_angle = angle_list[np.random.randint(0, 4)]
        else:
            set_angle = rotate

        img_M = cv2.getRotationMatrix2D((w / 2, h / 2), set_angle, 1)
        img_rotated = cv2.warpAffine(img, img_M, (w, h))

    return img_rotated


def Mirror(img, mirror='both'):
    if mirror is not None:
        if mirror == 'horizontal':
            set_flip = 0
        elif mirror == 'vertical':
            set_flip = 0
        elif mirror == 'both':
            set_flip = np.random.randint(0, 2)
        else:
            raise ValueError('The mirror setting "{}" is unknown'.format(mirror))
        img_m = cv2.flip(img, set_flip)
    else:
        img_m = img

    return img_m


def Substract_mean(img, mean_value):
    if mean_value is not None:
        img = img - mean_value
    return img


def Scaling(img, scale):
    if scale is not None:
        img = img * scale
    return img


# -----------------------------------------------------------------------------#

## Augment images
class Augmenter:
    """
    Conduct data augmentation on images

    Parameters
    ----------
    new_size : list of resized height and width (e.g. [h,w])
    delta_size : list of delta size for resizing (e.g. [delta_h, delta_w])
    crop_size : list of croped image size (e.g. [crop_h, crop_w])
    rotate : int for setting rotation angle
    mirror : string for setting mirror (horizontal, vertical, both)
    mean : list of channel mean or ndarray of mean image
    scale : int or float to scale each pixel
    """

    def __init__(self, new_size, delta_size, crop_size, crop_type, rotate, mirror, mean_value, scale):
        self.new_size = new_size
        self.delta_size = delta_size
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.rotate = rotate
        self.mirror = mirror
        self.mean_value = mean_value
        self.scale = scale

    def _Augment_each(self, img):
        img_aug = Resize(img, self.new_size, self.delta_size)
        img_aug = Substract_mean(img_aug, self.mean_value)
        img_aug = Scaling(img_aug, self.scale)
        img_aug = Crop(img_aug, self.crop_size, self.crop_type)
        img_aug = Rotate(img_aug, self.rotate)
        img_aug = Mirror(img_aug, self.mirror)

        return img_aug

    def Augment(self, images):
        num_cpus = multiprocessing.cpu_count()
        pool = ThreadPool(num_cpus)
        images = pool.map(self._Augment_each, images)
        pool.close()
        pool.join()

        return images

