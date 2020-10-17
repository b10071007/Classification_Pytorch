import torch
import torch.nn as nn

# num_blocks = [3, 4, 6, 3]
# conv_channels = [64, 128, 256, 512]

class ResNextBlock_bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride = 1, cardinality=16):
        super(ResNextBlock_bottleneck,self).__init__()
        self.expansion = 4
        self.all_channels = out_channel * cardinality
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
            nn.Conv2d(in_channel, self.all_channels, kernel_size=1, stride=1, bias=self.bias),
            nn.BatchNorm2d(self.all_channels),
            nn.ReLU(inplace=True),
            # conv 3x3
            nn.Conv2d(self.all_channels, self.all_channels, kernel_size=3, stride=self.stride, padding=1, bias=self.bias, groups=cardinality),
            nn.BatchNorm2d(self.all_channels),
            nn.ReLU(inplace=True),
            # conv 1x1
            nn.Conv2d(self.all_channels, out_channel*self.expansion, kernel_size=1, stride=1, bias=self.bias),
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

def ResGroup(block, in_channel, out_channel, num_blocks=2, stride=2, cardinality=32):

    layers = []
    layers.append(block(in_channel, out_channel, stride, cardinality))
    in_channel_current = out_channel * block.expansion
    for i in range(1, num_blocks):
        layers.append(block(in_channel_current, out_channel, 1, cardinality))

    return nn.Sequential(*layers)


class ResNeXt(nn.Module):
    def __init__(self, block, num_classes, num_blocks=[3, 4, 6, 3], conv_channels=[64, 128, 256, 512], cardinality=32, 
                 small_input=True, init_weights=True):
        super(ResNeXt, self).__init__()

        if small_input:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, conv_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(conv_channels[0]),
                nn.ReLU(inplace=True),
            )
            self.group4 = nn.Sequential()
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, conv_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(conv_channels[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
            self.group4 = ResGroup(block, conv_channels[2]*block.expansion, conv_channels[3], num_blocks[3])

        self.group1 = ResGroup(block, conv_channels[0], conv_channels[0], num_blocks[0], 1, cardinality)
        self.group2 = ResGroup(block, conv_channels[0]*block.expansion, conv_channels[1], num_blocks[1], cardinality)
        self.group3 = ResGroup(block, conv_channels[1]*block.expansion, conv_channels[2], num_blocks[2], cardinality)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(conv_channels[-1]*block.expansion, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)

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

# for Cifar
def Build_ResNeXt29_8x64d(num_classes=10, init_weights=True):
    return ResNeXt(ResNextBlock_bottleneck, num_classes, num_blocks=[3, 3, 3], conv_channels=[64, 128, 256], cardinality=8, 
                   small_input=True, init_weights=True)
def Build_ResNeXt29_16x64d(num_classes=10, init_weights=True):
    return ResNeXt(ResNextBlock_bottleneck, num_classes, num_blocks=[3, 3, 3], conv_channels=[64, 128, 256], cardinality=16, 
                   small_input=True, init_weights=True)

# for ImageNet
