currentPath=$PWD
rootPath=D:/Dataset/Classification/cifar10
model_name=DenseNet100_k12

# #-----------------------------------------------------------------------------------#

cd E:/Coding/Pytorch/project/Classification_Pytorch

python Train.py \
    -img_dir ${rootPath}/_Images/ \
    -fList_train ${rootPath}/train_all.txt \
    -fList_val ${rootPath}/test.txt \
    -num_classes 10 \
    -out_dir ./weights/${model_name}/test/ \
    \
    -gpu_id 0 \
    -model_name ${model_name} \
    -batch_size_train 64 \
    -batch_size_val 100 \
    -epochs 300 \
    -val_epochs 10 \
    -display_interval 200 \
    \
    -resize 48 48 \
    -random_resize \
    -random_resize_scale 0.8 1.2 \
    -random_resize_ratio 0.75 1.33 \
    -random_flip_H \
    -random_flip_V \
    -random_rotation 0 90 180 270 \
    -color_jitter_factor 0.5 0.5 0.5 0.3 \
    -random_crop_size 32 32 \
    -resize_val 48 48 \
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
    
# #-----------------------------------------------------------------------------------#

cd ${currentPath}