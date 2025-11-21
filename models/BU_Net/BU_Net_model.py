from .BU_Net_layers import *
from torch.utils import checkpoint

class BU_Net(nn.Module):
    
    def __init__(self, n_channels):
        super(BU_Net, self).__init__()
        self.n_channels = n_channels

        # Encoder
        self.in_conv = TripleConv(n_channels, 64, stride=1)
        self.down1 = Down(64, 128, stride=1)
        self.down2 = Down(128, 256, stride=1)
        self.down3 = Down(256, 512, stride=1)
        self.down4 = Down(512, 1024, stride=1)
        self.down5 = Down(1024, 2048, stride=1)

        # Bottleneck
        self.bottleneck = TripleConv(2048, 2048)
        
        # Decoder
        self.up1 = Up(2048 + 1024, 1024, stride=1)  # 2048 (bottleneck) + 1024 (skip from down4)
        self.up2 = Up(1024 + 512, 512, stride=1)    # 1024 (up1) + 512 (skip from down3)
        self.up3 = Up(512 + 256, 256, stride=1)     # 512 (up2) + 256 (skip from down2)
        self.up4 = Up(256 + 128, 128, stride=1)     # 256 (up3) + 128 (skip from down1)
        self.up5 = Up(128 + 64, 64, stride=1)       # 128 (up4) + 64 (skip from in_conv)

        # Output
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        
        
    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.bottleneck(x6)
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x4)
        x10 = self.up3(x9, x3)
        x11 = self.up4(x10, x2)
        x12 = self.up5(x11, x1)
        x13 = self.out_conv(x12)
        return x13
    
    def use_checkpoint(self):
        # for optimizing memory usage during training
        self.in_conv = checkpoint(self.in_conv)
        self.down1 = checkpoint(self.down1)
        self.down2 = checkpoint(self.down2)
        self.down3 = checkpoint(self.down3)
        self.down4 = checkpoint(self.down4)
        self.down5 = checkpoint(self.down5)
        self.bottleneck = checkpoint(self.bottleneck)
        self.up1 = checkpoint(self.up1)
        self.up2 = checkpoint(self.up2)
        self.up3 = checkpoint(self.up3)
        self.up4 = checkpoint(self.up4)
        self.up5 = checkpoint(self.up5)
        self.out_conv = checkpoint(self.out_conv)