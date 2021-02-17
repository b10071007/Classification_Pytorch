set currentPath=%CD%
set imgPath=D:/Dataset/Classification/cifar10
set programDir=E:/Coding/Pytorch/project/Classification_Pytorch
set model_name=ResNet110

@REM #-----------------------------------------------------------------------------------#

python %programDir%/Train.py ^
    -img_dir %imgPath%/_Images/ ^
    -fList_train %imgPath%/train_all.txt ^
    -fList_val %imgPath%/test.txt ^
    -num_classes 10 ^
    -out_dir ./debug/model/%model_name%/r2 ^
    ^
    -gpu_id 0 ^
    -model_name %model_name% ^
    -stride_times 2 ^
    -batch_size_train 128 ^
    -batch_size_val 100 ^
    -epochs 300 ^
    -val_epochs 10 ^
    -display_interval 200 ^
    ^
    -resize 36 36 ^
    -random_flip_H ^
    -random_flip_V ^
    -random_rotation 0 90 180 270 ^
    -random_crop_size 32 32 ^
    -resize_val 32 32 ^
    -center_crop_val 32 32 ^
    ^
    -channel_mean 0.4914 0.4822 0.4465 ^
    -channel_std 0.2023 0.1994 0.2010 ^
    ^
    -base_lr 0.1 ^
    -gamma 0.1 ^
    -lr_decay_steps 150 225 ^
    -weight_decay 0.0005 ^
    -warm_epoch 3
    
@REM #-----------------------------------------------------------------------------------#

cd %currentPath%