from .Nano_U_layers import *

class Nano_A(nn.Module):
    
    def __init__(self, n_channels):
        super(Nano_A, self).__init__()
        self.n_channels = n_channels

#######################################################
# per mantenere uno scaling (vedi EfficientNet) usare:
# psi = -5 (rispetto a BU_Net)

        # Encoder
        self.in_conv = DoubleConv(n_channels, 32, stride=1)
        self.down1 = Down(32, 64, stride=1)
        self.down2 = Down(64, 128, stride=1)
        self.down3 = Down(128, 256, stride=1)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 256)

        # Decoder
        self.up1 = Up(256, 128, stride=1)
        self.up2 = Up(128, 64, stride=1)
        self.up3 = Up(64, 32, stride=1)

        # Output
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)
        
        
    def forward(self, x):
        x = self.in_conv(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out_conv(x)
        return x
