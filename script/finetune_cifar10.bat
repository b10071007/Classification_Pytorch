set currentPath=%CD%
set imgPath=D:/Dataset/Classification/cifar10
set programDir=E:/Coding/Pytorch/project/Classification_Pytorch
set model_name=ResNet152_v2

@REM #-----------------------------------------------------------------------------------#

python %programDir%/Train.py ^
    -img_dir %imgPath%/_Images/ ^
    -fList_train %imgPath%/train_all.txt ^
    -fList_val %imgPath%/test.txt ^
    -num_classes 10 ^
    -out_dir ./debug/model/%model_name%/r1 ^
    ^
    -gpu_id 0 ^
    -model_name %model_name% ^
    -pretrained ./pretrained/ResNet152_v2_ImgNet_70.2.pth ^
    -stride_times 4 ^
    -batch_size_train 128 ^
    -batch_size_val 100 ^
    -epochs 100 ^
    -val_epochs 10 ^
    -display_interval 200 ^
    ^
    -resize 34 34 ^
    -random_flip_H ^
    -random_crop_size 32 32 ^
    -resize_val 34 34 ^
    -center_crop_val 32 32 ^
    ^
    -channel_mean 0.4914 0.4822 0.4465 ^
    -channel_std 0.2023 0.1994 0.2010 ^
    ^
    -base_lr 0.01 ^
    -gamma 0.1 ^
    -lr_decay_steps 20 50 80 ^
    -weight_decay 0.0005 ^
    -warm_epoch 1

    @REM -random_flip_V ^
    @REM -random_rotation 0 90 180 270 ^
    
@REM #-----------------------------------------------------------------------------------#

cd %currentPath%