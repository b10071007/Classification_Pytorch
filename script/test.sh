currentPath=$PWD
rootPath=D:/Dataset/Classification/cifar10
model_name=DenseNet100_k12

# #-----------------------------------------------------------------------------------#

cd E:/Coding/Pytorch/project/Classification_Pytorch

python Test.py \
    -img_dir ${rootPath}/_Images/ \
    -fList_test ${rootPath}/test.txt \
    -num_classes 10 \
    -out_dir ./weights/${model_name}/test/ \
    -model_path None \
    -channel_mean 0.4914 0.4822 0.4465 \
    -channel_std 0.2023 0.1994 0.2010 \
    -gpu_id 0 \
    -model_name ${model_name} \
    -batch_size_test 100 \
    -display_interval 100
    
# #-----------------------------------------------------------------------------------#

cd ${currentPath}