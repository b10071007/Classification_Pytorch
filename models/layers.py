
import torch.nn as nn



class Conv(nn.Module):
    """ 
    Highr API for Convolution Layer.
    
    Args:
    ---
        c1 (int): input channels
        c2 (int): output channels
        k (int): kernel size
        s (int): stride
        p (int): padding
        g (int): groups
        bias (bool): with bias or not
        bn (bool): with batch normalization or not
        act (bool): with relu or not
    """
    def __init__(self, c1, c2, k=3, s=1, p=0, g=1, bias=True, bn=False, act=False):  
        super(Conv, self).__init__()
        layer_list = []
        # conv
        layer_list.append(nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=p, groups=g, bias=bias))
        # batch normalization
        if bn:
            layer_list.append(nn.BatchNorm2d(c2))
        # relu
        if act:
            layer_list.append(nn.ReLU(inplace=True))
            
        self.conv = nn.Sequential(*layer_list)

    def forward(self, x):
        return self.conv(x)
    
#--------------------------------------------------------------------------------------------------------#  
    
if __name__ == '__main__':
    conv = Conv(3,32)
    print(conv)
    
