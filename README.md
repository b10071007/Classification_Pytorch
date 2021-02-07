# Classification_Pytorch
Classification on cifar-10 or any other dataset with Pytorch

### Supported Architectures
- VGG
- ResNet_v1
- ResNet_v2
- WideResNet
- DenseNet

### Supported Training tricks
- Step Learning Rate Decay
- Learning Rate Warm Up
- Model Selection based on the validation accuracy
- Data Augmentation
    - Resize, Random Resize, Random Crop, Horizontal and Vertical Flip, Random Rotation, Color Jittering


### Other tools
- [ArgsGenerator.py](https://github.com/b10071007/Classification_Pytorch/blob/master/tools/ArgsGenerator.py) : Parse [train.bat](https://github.com/b10071007/Classification_Pytorch/blob/master/script/train.bat) & [test.bat](https://github.com/b10071007/Classification_Pytorch/blob/master/script/test.bat) to generate [args_train.txt](https://github.com/b10071007/Classification_Pytorch/blob/master/.vscode/args_train.txt) & [args_test.txt](https://github.com/b10071007/Classification_Pytorch/blob/master/.vscode/args_test.txt) for vscode debugger setting [(launch.json)](https://github.com/b10071007/Classification_Pytorch/blob/master/.vscode/launch.json)
- [ScriptConverter.py](https://github.com/b10071007/Classification_Pytorch/blob/master/tools/ScriptConverter.py) : Conversion between batch script and shell script for executing Train.py and Test.py (e.g. convert [train.bat](https://github.com/b10071007/Classification_Pytorch/blob/master/script/train.bat) into [train.sh](https://github.com/b10071007/Classification_Pytorch/blob/master/script/train.sh))

