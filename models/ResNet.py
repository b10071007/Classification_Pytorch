import torch
import torch.nn as nn

# num_blocks = [3, 4, 6, 3]
# conv_channels = [64, 128, 256, 512]

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
    def __init__(self, block, num_classes, num_blocks=[3, 4, 6, 3], conv_channels=[64, 128, 256, 512], init_weights=True):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            # nn.Conv2d(3, conv_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.Conv2d(3, conv_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True),
        )
        # self.maxpool =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Sequential()

        self.group1 = ResGroup(block, conv_channels[0], conv_channels[0], num_blocks[0], 1)
        self.group2 = ResGroup(block, conv_channels[0]*block.expansion, conv_channels[1], num_blocks[1])
        self.group3 = ResGroup(block, conv_channels[1]*block.expansion, conv_channels[2], num_blocks[2])
        self.group4 = ResGroup(block, conv_channels[2]*block.expansion, conv_channels[3], num_blocks[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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

def Build_ResNet18(num_classes=10, init_weights=True):
    return ResNet(ResBlock, num_classes, num_blocks=[2, 2, 2, 2], init_weights=True)

def Build_ResNet34(num_classes=10, init_weights=True):
    return ResNet(ResBlock, num_classes, num_blocks=[3, 4, 6, 3], init_weights=True)

def Build_ResNet50(num_classes=10, init_weights=True):
    return ResNet(ResBlock_bottleneck, num_classes, num_blocks=[3, 4, 6, 3], init_weights=True)

def Build_ResNet101(num_classes=10, init_weights=True):
    return ResNet(ResBlock_bottleneck, num_classes, num_blocks=[3, 4, 23, 3], init_weights=True)

def Build_ResNet152(num_classes=10, init_weights=True):
    return ResNet(ResBlock_bottleneck, num_classes, num_blocks=[3, 8, 36, 3], init_weights=True)