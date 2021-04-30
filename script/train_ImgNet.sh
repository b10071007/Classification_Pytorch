currentPath=$PWD
#imgPath=D:/Dataset/Classification/cifar10
imgPath=/data2/Dataset/ImageNet
#programDir=E:/Coding/Pytorch/project/Classification_Pytorch
programDir=/HDD/Classification/project/Classification_Pytorch
model_name=ResNet101_v2

#-----------------------------------------------------------------------------------#

python ${programDir}/Train.py \
    -img_dir ${imgPath}/_Images_mix/ \
    -fList_train ${imgPath}/_train_map.txt \
    -fList_val ${imgPath}/_val_map.txt \
    -num_classes 1000 \
    -out_dir ./../model/${model_name}/like_paper/ \
    -num_workers 16 \
    \
    -gpu_id 0,1,2,3 \
    -model_name ${model_name} \
    -stride_times 5 \
    -batch_size_train 256 \
    -batch_size_val 250 \
    -epochs 90 \
    -val_epochs 2 \
    -display_interval 500 \
    \
    -random_resize \
    -random_resize_scale 0.08 1.0 \
    -random_resize_ratio 0.75 1.33 \
    -random_flip_H \
    -random_crop_size 224 224 \
    -resize_val 256 256 \
    -center_crop_val 224 224 \
    \
    -channel_mean 0.485 0.456 0.406 \
    -channel_std 0.229 0.224 0.225 \
    \
    -base_lr 0.1 \
    -gamma 0.1 \
    -lr_decay_steps 30 60 \
    -weight_decay 0.0001 \
    -warm_epoch 1

    #-resize 256 256 \
    #-random_flip_V \
    #-random_rotation 0 90 180 270 \
    #-random_resize_scale 0.5 0.625 \
    #-color_jitter_factor 0.5 0.5 0.5 0.3 \

    
#-----------------------------------------------------------------------------------#

python ${programDir}/Train.py \
    -img_dir ${imgPath}/_Images_mix/ \
    -fList_train ${imgPath}/_train_map.txt \
    -fList_val ${imgPath}/_val_map.txt \
    -num_classes 1000 \
    -out_dir ./../model/${model_name}/like_paper_add_colorJitter/ \
    \
    -gpu_id 0,1,2,3 \
    -model_name ${model_name} \
    -stride_times 5 \
    -batch_size_train 256 \
    -batch_size_val 250 \
    -epochs 90 \
    -val_epochs 2 \
    -display_interval 500 \
    \
    -random_resize \
    -random_resize_scale 0.08 1.0 \
    -random_resize_ratio 0.75 1.33 \
    -random_flip_H \
    -random_crop_size 224 224 \
    -resize_val 256 256 \
    -center_crop_val 224 224 \
    -color_jitter_factor 0.5 0.5 0.5 0.3 \
    \
    -channel_mean 0.485 0.456 0.406 \
    -channel_std 0.229 0.224 0.225 \
    \
    -base_lr 0.1 \
    -gamma 0.1 \
    -lr_decay_steps 30 60 \
    -weight_decay 0.0001 \
    -warm_epoch 1

    #-resize 256 256 \
    #-random_flip_V \
    #-random_rotation 0 90 180 270 \
    #-random_resize_scale 0.5 0.625 \

    
# #-----------------------------------------------------------------------------------#

cd ${currentPath}
