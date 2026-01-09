import torch.nn as nn

class DoubleConv(nn.Module):
        
    def __init__(self, in_channels, out_channels, mid_channels=None, stride=1):
        super().__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            # conv1
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            #conv2
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

    
class Down(nn.Module):
    """Downscaling with avgpooling then double conv"""
    
    def __init__(self, in_channels, out_channles, stride=1):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channles, stride=stride)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)
  
    
class Up(nn.Module):
    """Upscaling then double conv without skip connections"""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, stride=stride)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x
    
class OutConv(nn.Module):
    """Final convolution"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
    
    def forward(self, x):
        return self.conv(x)
