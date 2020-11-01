set currentPath=%CD%
set rootPath=D:/Dataset/Classification/cifar10
set model_name=DenseNet100_k12

@REM #-----------------------------------------------------------------------------------#

cd E:/Coding/Pytorch/project/Classification_Pytorch

python Train.py ^
    -img_dir %rootPath%/_Images/ ^
    -fList_train %rootPath%/train_all.txt ^
    -fList_val %rootPath%/test.txt ^
    -num_classes 10 ^
    -out_dir ./weights/%model_name%/test/ ^
    -channel_mean 0.4914 0.4822 0.4465 ^
    -channel_std 0.2023 0.1994 0.2010 ^
    -gpu_id 0 ^
    -model_name %model_name% ^
    -batch_size_train 64 ^
    -batch_size_val 100 ^
    -epochs 300 ^
    -val_epochs 10 ^
    -display_interval 200 ^
    -base_lr 0.1 ^
    -gamma 0.1 ^
    -lr_decay_steps 150 225 ^
    -weight_decay 0.0005 ^
    -warm_epoch 1
    
@REM #-----------------------------------------------------------------------------------#

cd %currentPath%