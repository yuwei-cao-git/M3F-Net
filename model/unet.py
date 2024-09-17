import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=12, out_channels=9):
        super(UNet, self).__init__()

        # Encoder
        self.enc_conv0 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.enc_conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1)

        # Decoder
        self.dec_conv3 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_conv0 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        # Maxpool and upsample
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.enc_conv0(x))
        x2 = self.pool(x1)
        x2 = F.relu(self.enc_conv1(x2))
        x3 = self.pool(x2)
        x3 = F.relu(self.enc_conv2(x3))
        x4 = self.pool(x3)
        x4 = F.relu(self.enc_conv3(x4))

        # Decoder
        x = self.up(x4)
        x = F.relu(self.dec_conv3(x))
        x = self.up(x)
        x = F.relu(self.dec_conv2(x))
        x = self.up(x)
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv0(x))  # Output layer without activation

        return x
