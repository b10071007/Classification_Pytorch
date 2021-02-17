from torch import rand
import torch.nn as nn

import sys
sys.path.append(".")
from models.layers import Conv

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channel, out_channel, stride = 1):
        super(ResBlock,self).__init__()
        self.bias = False
       
        self.downsample_shortcut =  nn.Sequential()
        self.stride = stride

        if (in_channel!=out_channel*self.expansion):
            self.downsample_shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=self.stride, bias=self.bias),
                nn.BatchNorm2d(out_channel*self.expansion),
            )

        self.convs = nn.Sequential(
            # conv 3x3
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=self.stride, padding=1, bias=self.bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # conv 3x3
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.BatchNorm2d(out_channel),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        x = self.convs(x)
        shortcut = self.downsample_shortcut(shortcut)
        x = x + shortcut
        x = self.relu(x)

        return x

class ResBlock_bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride = 1):
        super(ResBlock_bottleneck,self).__init__()
        self.expansion = 4
        self.bias = False
       
        self.downsample_shortcut =  nn.Sequential()
        self.stride = stride

        if (in_channel!=out_channel*self.expansion):
            self.downsample_shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*self.expansion, kernel_size=1, stride=self.stride, bias=self.bias),
                nn.BatchNorm2d(out_channel*self.expansion),
            )

        self.convs = nn.Sequential(
            # conv 1x1
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=self.bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # conv 3x3
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=self.stride, padding=1, bias=self.bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # conv 1x1
            nn.Conv2d(out_channel, out_channel*self.expansion, kernel_size=1, stride=1, bias=self.bias),
            nn.BatchNorm2d(out_channel*self.expansion),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        x = self.convs(x)
        shortcut = self.downsample_shortcut(shortcut)
        x = x + shortcut
        x = self.relu(x)

        return x

def ResGroup(block, in_channel, out_channel, num_blocks=2, stride=2):

    layers = []
    layers.append(block(in_channel, out_channel, stride))
    in_channel_current = out_channel * block.expansion
    for i in range(1, num_blocks):
        layers.append(block(in_channel_current, out_channel, 1))

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, block, num_classes, num_blocks=[3, 4, 6, 3], conv_channels=[64, 128, 256, 512], 
                 stride_times=5, init_weights=True):
        super(ResNet, self).__init__()

        # Normal input size (for ImageNet; e.g. 224, 256, 288)
        if stride_times==5:
            self.conv1 = Conv(c1=3, c2=conv_channels[0], k=7, s=2, p=3, g=1, bias=False, bn=True, act=True)
            self.maxpool =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Small input size (e.g. 112, 128, 144)
        elif stride_times==4:
            self.conv1 = Conv(c1=3, c2=conv_channels[0], k=7, s=1, p=3, g=1, bias=False, bn=True, act=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Tiny input size (e.g. 56, 64, 72)
        elif stride_times==3:
            self.conv1 = Conv(c1=3, c2=conv_channels[0], k=3, s=1, p=1, g=1, bias=False, bn=True, act=True)
            self.maxpool = nn.Sequential()
        # Extreme tiny input size (for Cifar; e.g. 28, 32, 36)
        elif stride_times==2:
            self.conv1 = Conv(c1=3, c2=conv_channels[0], k=3, s=1, p=1, g=1, bias=False, bn=True, act=True)
            self.maxpool = nn.Sequential()  
            # no stride in first conv and remove group4

        self.group1 = ResGroup(block, conv_channels[0], conv_channels[0], num_blocks[0], 1)
        self.group2 = ResGroup(block, conv_channels[0]*block.expansion, conv_channels[1], num_blocks[1])
        self.group3 = ResGroup(block, conv_channels[1]*block.expansion, conv_channels[2], num_blocks[2])
        
        if stride_times==2: # remove group4 for stride_times=2
            self.group4 = nn.Sequential()
        else:
            self.group4 = ResGroup(block, conv_channels[2]*block.expansion, conv_channels[3], num_blocks[3])
                    
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        if stride_times==2: # use conv_channels[2] because group4 was removed
            self.fc = nn.Linear(conv_channels[2]*block.expansion, num_classes)
        else:
            self.fc = nn.Linear(conv_channels[3]*block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.group4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

#---------------------------------------------------------------------------------------------------------------#

def Build_ResNet18(num_classes=10, stride_times=5, init_weights=True):
    return ResNet(ResBlock, num_classes, num_blocks=[2, 2, 2, 2], stride_times=stride_times, init_weights=True)

def Build_ResNet34(num_classes=10, stride_times=5, init_weights=True):
    return ResNet(ResBlock, num_classes, num_blocks=[3, 4, 6, 3], stride_times=stride_times, init_weights=True)

def Build_ResNet50(num_classes=10, stride_times=5, init_weights=True):
    return ResNet(ResBlock_bottleneck, num_classes, num_blocks=[3, 4, 6, 3], stride_times=stride_times, init_weights=True)

def Build_ResNet101(num_classes=10, stride_times=5, init_weights=True):
    return ResNet(ResBlock_bottleneck, num_classes, num_blocks=[3, 4, 23, 3], stride_times=stride_times, init_weights=True)

def Build_ResNet152(num_classes=10, stride_times=5, init_weights=True):
    return ResNet(ResBlock_bottleneck, num_classes, num_blocks=[3, 8, 36, 3], stride_times=stride_times, init_weights=True)

# for cifar-10
def Build_ResNet110(num_classes=10, stride_times=5, init_weights=True):
    assert stride_times==2, "stride_times of ResNet110_v2 (for cifar) should be 2"
    return ResNet(ResBlock, num_classes, num_blocks=[18, 18, 18], stride_times=stride_times, conv_channels=[16, 32, 64], init_weights=True)

#---------------------------------------------------------------------------------------------------------------#  

# for testing
if __name__ == '__main__':
    model = Build_ResNet18(num_classes=10, stride_times=3, init_weights=True)
    img = rand(1, 3, 32, 32)
    output = model.forward(img)
    print(model)
    
    model = Build_ResNet18(num_classes=10, stride_times=2, init_weights=True)
    img = rand(1, 3, 32, 32)
    output = model.forward(img)
    print(model)
        
    model = Build_ResNet101(num_classes=10, stride_times=4, init_weights=True)
    img = rand(1, 3, 128, 128)
    output = model.forward(img)
    print(model)
    
    model = Build_ResNet110(num_classes=10, stride_times=2, init_weights=True)
    img = rand(1, 3, 32, 32)
    output = model.forward(img)
    print(model)
    
    