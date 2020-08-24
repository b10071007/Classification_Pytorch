import torch
import torch.nn as nn
# import torch.nn.functional as F

def conv_block(in_channel, out_channel, num_conv=2, kernel_size=3, pool_size=2):
    
    layers = []
    for i in range(num_conv):
        if i==0:
            layers += [nn.Conv2d(in_channel, out_channel, kernel_size, 1, 1)]
        else:
            layers += [nn.Conv2d(out_channel, out_channel, kernel_size, 1, 1)]
        layers += [nn.ReLU(inplace=True)]
        
    layers += [nn.MaxPool2d(pool_size, pool_size)]
    
    return layers

def fc_layer(in_channel, out_channel):
    layers = [nn.Linear(in_channel, out_channel), nn.ReLU(inplace=True)]

    return layers

def create_feature(num_convs = [2, 2, 3, 3, 3], 
                   conv_channels = [64, 128, 256, 512, 512]):
    layers = []
    layers += conv_block(3, conv_channels[0], num_convs[0])
    layers += conv_block(conv_channels[0], conv_channels[1], num_convs[1])
    layers += conv_block(conv_channels[1], conv_channels[2], num_convs[2])
    layers += conv_block(conv_channels[2], conv_channels[3], num_convs[3])
    layers += conv_block(conv_channels[3], conv_channels[4], num_convs[4])
    
    return layers

class VGG(nn.Module):
    def __init__(self, feature, num_classes, init_weights, fc_channels):
        super(VGG, self).__init__()
        
        self.feature = nn.Sequential(*feature)

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            *fc_layer(7*7*fc_channels[0], fc_channels[1]),
            nn.Dropout(0.5),
            *fc_layer(fc_channels[1], fc_channels[2]),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
            )
        
        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
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

         
def VGG16(num_classes=10, init_weights=True, fc_channels=[512, 4096, 4096]):
    return VGG(create_feature([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
               num_classes, init_weights, fc_channels)

def VGG19(num_classes=10, init_weights=True, fc_channels=[512, 4096, 4096]):
    return VGG(create_feature([2, 2, 4, 4, 4], [64, 128, 256, 512, 512]),
               num_classes, init_weights, fc_channels)

# num_classes = 10
# init_weights = True

# model = VGG16(num_classes, init_weights)
