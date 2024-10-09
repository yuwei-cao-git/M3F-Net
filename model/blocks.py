import torch
import torch.nn as nn

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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)  # Adjust channel dimensions if necessary

    def forward(self, x):
        identity = self.skip(x)  # Skip connection
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity  # Add skip connection to the output
        return self.relu(out)