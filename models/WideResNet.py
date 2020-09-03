import torch
import torch.nn as nn
import torch.nn.functional as F

class WideResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, dropout=0):
        super(WideResBlock,self).__init__()
        self.bias = False
       
        self.downsample_shortcut =  nn.Sequential()
        self.stride = stride

        if (in_channel!=out_channel):
            self.downsample_shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=self.stride, bias=self.bias)
            )

        if dropout>0:
            self.convs = nn.Sequential(
                # conv 3x3
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=self.stride, padding=1, bias=self.bias),
                # conv 3x3
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias),
            )
        else:
            self.convs = nn.Sequential(
                # conv 3x3
                nn.BatchNorm2d(in_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=self.stride, padding=1, bias=self.bias),
                # conv 3x3
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                F.dropout(p=self.dropout, training=self.training),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=self.bias),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x

        x = self.convs(x)
        shortcut = self.downsample_shortcut(shortcut)
        x = x + shortcut

        return x

def WideResGroup(in_channel, out_channel, N=4, stride=1, dropout=0):

    layers = []
    layers.append(WideResBlock(in_channel, out_channel, stride, dropout))
    for _ in range(1, N):
        layers.append(WideResBlock(out_channel, out_channel, 1, dropout))

    return nn.Sequential(*layers)


class WideResNet(nn.Module):
    def __init__(self, num_classes, N=4, k=4, base_channel=16, init_weights=True, dropout=0.3):
        super(WideResNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channel, kernel_size=3, stride=1, padding=1, bias=False),
        )
        # self.maxpool =  nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool = nn.Sequential()

        self.group1 = WideResGroup(in_channel=base_channel, out_channel=base_channel*k, N=N, stride=1, dropout=dropout)
        self.group2 = WideResGroup(in_channel=base_channel*k, out_channel=base_channel*2*k, N=N, stride=2, dropout=dropout)
        self.group3 = WideResGroup(in_channel=base_channel*2*k, out_channel=base_channel*4*k, N=N, stride=2, dropout=dropout)
        
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

# WRN-28-4
def Build_WRN_N4_k4(num_classes=10, init_weights=True):
    return WideResNet(num_classes, N=4, k=4, init_weights=True)
# WRN-28-10
def Build_WRN_N4_k10(num_classes=10, init_weights=True):
    return WideResNet(num_classes, N=4, k=10, init_weights=True)
# WRN-28-12
def Build_WRN_N4_k12(num_classes=10, init_weights=True):
    return WideResNet(num_classes, N=4, k=12, init_weights=True)
# WRN-40-12
def Build_WRN_N6_k10(num_classes=10, init_weights=True):
    return WideResNet(num_classes, N=6, k=10, init_weights=True)