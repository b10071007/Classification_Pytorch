import torch
import torch.nn as nn

# num_blocks = [3, 4, 6, 3]
# conv_channels = [64, 128, 256, 512]

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample = False):
        super(ResBlock,self).__init__()
        self.downsample = downsample
        self.bias = False
        
        if self.downsample:
            self.downsample_shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel*4, kernel_size=1, stride=2, bias=self.bias),
                nn.BatchNorm2d(out_channel*4)
            )
            self.stride_conv3x3 = 2
        else:
            self.stride_conv3x3 = 1


        self.convs = nn.Sequential(
            # conv 1x1
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=1, bias=self.bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # conv 3x3
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=self.stride_conv3x3, padding=1, bias=self.bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            # conv 1x1
            nn.Conv2d(out_channel, out_channel*4, kernel_size=1, stride=1, padding=1, bias=self.bias),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        x = self.convs(x)

        if self.downsample:
            shortcut = self.downsample_shortcut(shortcut)
            
        x = x + shortcut
        x = self.relu(x)

        return x

def ResGroup(in_channel, out_channel, num_blocks=2):

    in_channel_current = in_channel
    layers = []
    layers.append(ResBlock(in_channel, out_channel))
    for i in range(1, num_blocks):
        layers.append(ResBlock(in_channel_current, out_channel))
        in_channel_current = out_channel*4

    return nn.Sequential(*layers)


class ResNet(nn.Module):
    def __init__(self, num_classes, num_blocks=[3, 4, 6, 3], conv_channels=[64, 128, 256, 512], init_weights=True):
        super(ResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, conv_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            # nn.Conv2d(3, conv_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(conv_channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool =  nn.MaxPool2d(kernel_size=1, stride=1, padding=1)

        self.group1 = ResGroup(conv_channels[0], conv_channels[0], num_blocks[0])
        self.group2 = ResGroup(conv_channels[0], conv_channels[1], num_blocks[1])
        self.group3 = ResGroup(conv_channels[1], conv_channels[2], num_blocks[2])
        self.group4 = ResGroup(conv_channels[2], conv_channels[3], num_blocks[3])
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.maxpool(x)
        print(x.shape)
        x = self.group1(x)
        print(x.shape)
        x = self.group2(x)
        print(x.shape)
        x = self.group3(x)
        print(x.shape)
        x = self.group4(x)
        print(x.shape)

        x = self.avgpool(x)
        print(x.shape)
        x = self.fc(x)
        print(x.shape)
        exit()
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


def ResNet50(num_classes=10, init_weights=True):
    return ResNet(num_classes, num_blocks=[3, 4, 6, 3], conv_channels=[64, 128, 256, 512], init_weights=True)

def ResNet101(num_classes=10, init_weights=True):
    return ResNet(num_classes, num_blocks=[3, 4, 23, 3], conv_channels=[64, 128, 256, 512], init_weights=True)



# num_classes = 10
# init_weights = True

# model = ResNet50(num_classes, init_weights)
# print(model)