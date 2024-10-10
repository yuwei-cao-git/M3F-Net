import torch
import torch.nn as nn
import torch.nn.functional as F

""" Parts of the U-Net model """
# fusion s2 data
class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上
    
class MF(nn.Module):  # Multi-Feature (MF) module for seasonal attention-based fusion
    def __init__(self, channels=13, reduction=16):  # Each season has 13 channels
        super(MF, self).__init__()
        # Channel attention for each season (spring, summer, autumn, winter)
        self.channels=channels
        self.reduction=reduction
        self.mask_map_spring = nn.Conv2d(self.channels, 1, 1, 1, 0, bias=True)
        self.mask_map_summer = nn.Conv2d(self.channels, 1, 1, 1, 0, bias=True)
        self.mask_map_autumn = nn.Conv2d(self.channels, 1, 1, 1, 0, bias=True)
        self.mask_map_winter = nn.Conv2d(self.channels, 1, 1, 1, 0, bias=True)
        
        # Shared bottleneck layers for each season
        self.bottleneck_spring = nn.Conv2d(self.channels, 16, 3, 1, 1, bias=False)
        self.bottleneck_summer = nn.Conv2d(self.channels, 16, 3, 1, 1, bias=False)
        self.bottleneck_autumn = nn.Conv2d(self.channels, 16, 3, 1, 1, bias=False)
        self.bottleneck_winter = nn.Conv2d(self.channels, 16, 3, 1, 1, bias=False)
        
        # Final SE Block for channel attention across all seasons
        self.se = SE_Block(64, self.reduction)  # Since we have 4 seasons with 16 channels each, we get a total of 64 channels

    def forward(self, x):  # x is a list of 4 inputs (spring, summer, autumn, winter)
        spring, summer, autumn, winter = x  # Unpack the inputs

        # Apply attention maps
        spring_mask = torch.mul(self.mask_map_spring(spring).repeat(1, self.channels, 1, 1), spring)
        summer_mask = torch.mul(self.mask_map_summer(summer).repeat(1, self.channels, 1, 1), summer)
        autumn_mask = torch.mul(self.mask_map_autumn(autumn).repeat(1, self.channels, 1, 1), autumn)
        winter_mask = torch.mul(self.mask_map_winter(winter).repeat(1, self.channels, 1, 1), winter)

        # Apply bottleneck layers
        spring_features = self.bottleneck_spring(spring_mask)
        summer_features = self.bottleneck_summer(summer_mask)
        autumn_features = self.bottleneck_autumn(autumn_mask)
        winter_features = self.bottleneck_winter(winter_mask)

        # Concatenate features from all seasons
        combined_features = torch.cat([spring_features, summer_features, autumn_features, winter_features], dim=1)

        # Apply SE Block for channel-wise attention
        out = self.se(combined_features)

        return out

""" Parts of the U-Net model """
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
        
""" Parts of the ResU-Net model """
class BNAct(nn.Module):
    """Batch Normalization followed by an optional ReLU activation."""
    def __init__(self, num_features, act=True):
        super(BNAct, self).__init__()
        self.bn = nn.BatchNorm2d(num_features)
        self.act = act
        if self.act:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        if self.act:
            x = self.activation(x)
        return x
    
# ConvBlock module
class ConvBlock(nn.Module):
    """Convolution Block with BN and Activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.bn_act = BNAct(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        x = self.bn_act(x)
        x = self.conv(x)
        return x

# Stem module
class Stem(nn.Module):
    """Initial convolution block with residual connection."""
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_block = ConvBlock(out_channels, out_channels)
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn_act = BNAct(out_channels, act=False)
    
    def forward(self, x):
        conv = self.conv1(x)
        conv = self.conv_block(conv)
        shortcut = self.shortcut_conv(x)
        shortcut = self.bn_act(shortcut)
        output = conv + shortcut
        return output

# ResidualBlock module
class ResidualBlock(nn.Module):
    """Residual block with two convolutional layers and a shortcut connection."""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv_block1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv_block2 = ConvBlock(out_channels, out_channels)
        self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.bn_act = BNAct(out_channels, act=False)
    
    def forward(self, x):
        res = self.conv_block1(x)
        res = self.conv_block2(res)
        shortcut = self.shortcut_conv(x)
        shortcut = self.bn_act(shortcut)
        output = res + shortcut
        return output

# UpSampleConcat module
class UpSampleConcat(nn.Module):
    """Upsamples the input and concatenates with the skip connection."""
    def __init__(self):
        super(UpSampleConcat, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, xskip):
        #x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.up(x)
        x = torch.cat([x, xskip], dim=1)
        return x