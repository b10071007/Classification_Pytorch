import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseUnit(nn.Module):
    def __init__(self, in_channel, expansion, growth_rate, dropout=0):
        super(DenseUnit,self).__init__()
        self.bias = False
        self.inter_channel = expansion*growth_rate
        self.dropout = dropout
        
        self.preAct1 = nn.Sequential(
            nn.BatchNorm2d(in_channel), 
            nn.ReLU(inplace=True),
            )
        self.conv1 = nn.Conv2d(in_channel, self.inter_channel, kernel_size=1, stride=1, bias=self.bias)
        
        self.preAct2 = nn.Sequential(
            nn.BatchNorm2d(self.inter_channel), 
            nn.ReLU(inplace=True),
            )
        self.conv2 = nn.Conv2d(self.inter_channel, growth_rate, kernel_size=3, stride=1, padding=1, bias=self.bias)

    def forward(self, x):
        shortcut = x

        x = self.preAct1(x)
        x = self.conv1(x)
        x = self.preAct2(x)
        x = self.conv2(x)
        
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = torch.cat([x, shortcut], dim=1)

        return x

class Transition(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Transition, self).__init__()

        self.preAct1 = nn.Sequential(
            nn.BatchNorm2d(in_channel), 
            nn.ReLU(inplace=True),
            )
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.preAct1(x)
        x = self.conv1(x)
        x = self.avg_pool(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, num_classes, growth_rate = 12, compression = 0.5, expansion = 4, 
                 num_blocks=[6, 12, 24, 16], dropout=0, init_weights=True):
        super(DenseNet, self).__init__()
        
        self.growth_rate = growth_rate
        self.in_channel = self.growth_rate*2
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.dense1 = self.makeDenseBlock(expansion=expansion, N=num_blocks[0], dropout=dropout)
        self.transition1 = self.makeTransition(compression=compression)
        self.dense2 = self.makeDenseBlock(expansion=expansion, N=num_blocks[1], dropout=dropout)
        self.transition2 = self.makeTransition(compression=compression)
        self.dense3 = self.makeDenseBlock(expansion=expansion, N=num_blocks[2], dropout=dropout)

        self.bn_last = nn.BatchNorm2d(self.in_channel)
        self.relu_last = nn.ReLU(inplace=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(self.in_channel, num_classes)

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.dense3(x)

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


    def makeDenseBlock(self, expansion=4, N=4, dropout=0):

        layers = []
        for _ in range(N):
            layers.append(DenseUnit(self.in_channel, expansion, self.growth_rate, dropout))
            self.in_channel = self.in_channel + self.growth_rate

        return nn.Sequential(*layers)

    def makeTransition(self, compression):
        in_channel = self.in_channel
        out_channel = int(self.in_channel * compression)
        self.in_channel = out_channel
        return Transition(in_channel, out_channel)

#---------------------------------------------------------------------------------------------------------------#

# WRN-28-4
# def Build_DenseNet121_k12(num_classes=10, init_weights=True):
#     return DenseNet(num_classes, growth_rate = 12, compression = 0.5, expansion = 4, num_blocks=[6, 12, 24, 16], dropout=0, init_weights=True)

# def Build_DenseNet169_k12(num_classes=10, init_weights=True):
#     return DenseNet(num_classes, growth_rate = 12, compression = 0.5, expansion = 4, num_blocks=[6, 12, 32, 32], dropout=0, init_weights=True)

def Build_DenseNet100_k12(num_classes=10, init_weights=True):
    return DenseNet(num_classes, growth_rate = 12, compression = 0.5, expansion = 4, num_blocks=[16, 16, 16], dropout=0, init_weights=True)

def Build_DenseNet100_k24(num_classes=10, init_weights=True):
    return DenseNet(num_classes, growth_rate = 24, compression = 0.5, expansion = 4, num_blocks=[16, 16, 16], dropout=0, init_weights=True)

def Build_DenseNet190_k40(num_classes=10, init_weights=True):
    return DenseNet(num_classes, growth_rate = 40, compression = 0.5, expansion = 4, num_blocks=[31, 31, 31], dropout=0, init_weights=True)
