currentPath=$PWD
#imgPath=D:/Dataset/Classification/cifar10
imgPath=/data/Classification/Dataset/cifar10
#programDir=E:/Coding/Pytorch/project/Classification_Pytorch
programDir=/HDD/Classification/project/Classification_Pytorch
model_name=ResNet110_v2

# #-----------------------------------------------------------------------------------#

# no random resize, no color jitter, no resize, no rotate
python ${programDir}/Train.py \
    -img_dir ${imgPath}/_Images/ \
    -fList_train ${imgPath}/train_all.txt \
    -fList_val ${imgPath}/test.txt \
    -num_classes 10 \
    -out_dir ./../model/${model_name}/flipH_resize34/ \
    \
    -gpu_id 0 \
    -model_name ${model_name} \
    -stride_times 2 \
    -batch_size_train 128 \
    -batch_size_val 100 \
    -epochs 300 \
    -val_epochs 10 \
    -display_interval 200 \
    \
    -resize 34 34 \
    -random_flip_H \
    -random_crop_size 32 32 \
    -resize_val 34 34 \
    -center_crop_val 32 32 \
    \
    -channel_mean 0.4914 0.4822 0.4465 \
    -channel_std 0.2023 0.1994 0.2010 \
    \
    -base_lr 0.1 \
    -gamma 0.1 \
    -lr_decay_steps 150 225 \
    -weight_decay 0.0005 \
    -warm_epoch 1

    #-random_resize \
    #-random_resize_scale 0.8 1.2 \
    #-random_resize_ratio 0.75 1.33 \
    #-random_rotation 0 90 180 270 \
    #-random_flip_V \
    #-color_jitter_factor 0.5 0.5 0.5 0.3 \
    
# #-----------------------------------------------------------------------------------#

cd ${currentPath}
