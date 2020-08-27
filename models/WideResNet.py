import torch
import torch.nn as nn

class WideResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample = False):
        super(WideResBlock,self).__init__()
        self.downsample = downsample
        self.bias = False
        
        if self.downsample:
            self.downsample_shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=2, bias=self.bias),
                nn.BatchNorm2d(out_channel)
            )
            self.stride_down = 2
        else:
            self.stride_down = 1


        self.convs = nn.Sequential(
            # conv 3x3
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=self.stride_down, padding=1, bias=self.bias),
            # conv 3x3
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias),
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

def WideResGroup(in_channel, out_channel, N=4, first_downsample=True):

    layers = []
    layers.append(WideResBlock(in_channel, out_channel, first_downsample))
    for _ in range(1, N):
        layers.append(WideResBlock(out_channel, out_channel, False))

    return nn.Sequential(*layers)


class WideResNet(nn.Module):
    def __init__(self, num_classes, N=4, k=4, base_channel=16, init_weights=True):
        super(WideResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        # self.maxpool =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Sequential()

        self.group1 = WideResGroup(in_channel=base_channel, out_channel=base_channel*k, N=N, first_downsample=True)
        self.group2 = WideResGroup(in_channel=base_channel*k, out_channel=base_channel*2*k, N=N, first_downsample=True)
        self.group3 = WideResGroup(in_channel=base_channel*2*k, out_channel=base_channel*4*k, N=N, first_downsample=True)
        
        self.bn_last = nn.BatchNorm2d(base_channel*4*k)
        self.relu_last = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channel*4*k, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)

        x = self.relu_last(self.bn_last(x))
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

def WRN_N4_k4(num_classes=10, init_weights=True):
    return WideResNet(num_classes, N=4, k=4, init_weights=True)

def WRN_N4_k10(num_classes=10, init_weights=True):
    return WideResNet(num_classes, N=4, k=10, init_weights=True)

def WRN_N4_k12(num_classes=10, init_weights=True):
    return WideResNet(num_classes, N=4, k=12, init_weights=True)

# def WRN_N6_k4(num_classes=10, init_weights=True):
#     return WideResNet(num_classes, N=4, k=10, init_weights=True)