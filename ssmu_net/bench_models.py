"""
Benchmark models based on successful pixel_pixel experiments.
These models achieved 0.9 mIoU in previous experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = x.view(b, c, -1).mean(-1)  # (B, C)
        # Excitation: FC layers
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        # Scale
        return x * y.view(b, c, 1, 1)


class SpectralAttention(nn.Module):
    """Spectral attention module from successful deep.py"""
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, 1),
            nn.GELU(),
            nn.Conv2d(channels // 8, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.attn(x)


class ResidualBlock(nn.Module):
    """Residual block with optional stride for downsampling"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return F.gelu(self.conv(x) + self.shortcut(x))


class SpectralAttentionNet(nn.Module):
    """
    Spectral compression + attention model.
    Based on successful FTIRNet from deep.py (achieved high accuracy).
    """
    def __init__(self, in_channels=425, num_classes=8):
        super().__init__()
        
        # Spectral Compression: 425 -> 64 channels
        self.spectral_compress = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Encoder with spectral attention
        self.encoder = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            SpectralAttention(128),
            ResidualBlock(128, 256, stride=2),
            SpectralAttention(256),
            ResidualBlock(256, 512, stride=2),
            SpectralAttention(512),
        )
        
        # Decoder with upsampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # x: (B, 425, H, W)
        x = self.spectral_compress(x)  # (B, 64, H, W)
        x = self.encoder(x)  # (B, 512, H/8, W/8)
        x = self.decoder(x)  # (B, 64, H, W)
        return self.seg_head(x)  # (B, num_classes, H, W)


class DoubleConv(nn.Module):
    """Double convolution block for UNet"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SECompressorUNet(nn.Module):
    """
    UNet with Spectral Compressor + SE Blocks.
    Based on successful model from pixel_pixel/scripts_for_csf/unet.py
    """
    def __init__(self, in_channels=425, num_classes=8):
        super().__init__()
        
        # Spectral compression: 425 -> 64
        self.spectral_compress = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        # Encoder (downsampling)
        self.enc1 = nn.Sequential(DoubleConv(64, 64), SEBlock(64))
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(DoubleConv(64, 128), SEBlock(128))
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = nn.Sequential(DoubleConv(128, 256), SEBlock(256))
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = nn.Sequential(DoubleConv(256, 512), SEBlock(512))
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(DoubleConv(512, 1024), SEBlock(1024))
        
        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = nn.Sequential(DoubleConv(1024, 512), SEBlock(512))
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(DoubleConv(512, 256), SEBlock(256))
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(DoubleConv(256, 128), SEBlock(128))
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(DoubleConv(128, 64), SEBlock(64))
        
        # Final segmentation head
        self.seg_head = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # Spectral compression
        x = self.spectral_compress(x)  # (B, 64, H, W)
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.seg_head(dec1)


class SimpleSpectralNet(nn.Module):
    """
    Simplest possible model for debugging.
    Just spectral compression + direct segmentation.
    """
    def __init__(self, in_channels=425, num_classes=8):
        super().__init__()
        self.net = nn.Sequential(
            # Spectral compression
            nn.Conv2d(in_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Spatial processing
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Segmentation
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x):
        return self.net(x)


def get_benchmark_model(model_name='spectral_attention', in_channels=425, num_classes=8):
    """
    Factory function to get benchmark models.
    
    Args:
        model_name: One of ['spectral_attention', 'se_unet', 'simple']
        in_channels: Number of input spectral channels
        num_classes: Number of segmentation classes
    
    Returns:
        Model instance
    """
    models = {
        'spectral_attention': SpectralAttentionNet,
        'se_unet': SECompressorUNet,
        'simple': SimpleSpectralNet
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    
    return models[model_name](in_channels=in_channels, num_classes=num_classes)