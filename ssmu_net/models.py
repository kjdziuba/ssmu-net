"""
SSMU-Net: Sinc spectral front-end + Mamba SSM + 2D U-Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List

try:
    from mamba_ssm import Mamba
except ImportError as e:
    raise ImportError("mamba_ssm is required. `pip install mamba-ssm`") from e


class SincConv1d(nn.Module):
    """
    Learnable band-pass filters over the spectral axis.
    Cutoffs are parameterized in cm^-1 and mapped to digital frequency.
    """
    
    def __init__(self, 
                 out_channels: int = 32,
                 kernel_size: int = 129,
                 wn_min: float = 900.0,
                 wn_max: float = 1800.0,
                 init_bw: float = 80.0):
        """
        Args:
            out_channels: Number of filters to learn
            kernel_size: Size of the sinc kernel (must be odd)
            wn_min: Minimum wavenumber in cm^-1
            wn_max: Maximum wavenumber in cm^-1
            init_bw: Initial bandwidth in cm^-1
        """
        super().__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        
        self.out = out_channels
        self.ks = kernel_size
        self.wn_min, self.wn_max = float(wn_min), float(wn_max)
        
        # Create time indices centered at 0
        n = torch.arange(-(self.ks//2), self.ks//2 + 1).float()  # (K,)
        self.register_buffer("n", n)
        
        # Cache Hamming window as buffer
        self.register_buffer("win", torch.hamming_window(self.ks, periodic=False))
        
        # Initialize filters with biochemical priors and uniform filling
        centers = self._init_centers(out_channels)
        bw = torch.full((out_channels,), init_bw)
        
        self.f_low = nn.Parameter(centers - bw/2)
        self.f_high = nn.Parameter(centers + bw/2)
    
    def _init_centers(self, n_filters: int) -> torch.Tensor:
        """Initialize filter centers with biochemical priors"""
        biochemical_bands = [
            1655,  # Amide I (α-helix)
            1545,  # Amide II
            1740,  # Lipid ester
            1240,  # Amide III
            1080,  # Nucleic acid (phosphodiester)
            1450,  # CH2 bending
            1395,  # COO- symmetric stretch
            1160,  # C-O stretch
        ]
        
        # Filter bands to those within our range
        bands = [b for b in biochemical_bands if self.wn_min + 50 < b < self.wn_max - 50]
        
        if len(bands) >= n_filters:
            # If we have enough biochemical bands, use them
            centers = torch.tensor(bands[:n_filters], dtype=torch.float32)
        else:
            # Mix biochemical bands with uniform sampling
            bio_centers = torch.tensor(bands, dtype=torch.float32)
            n_uniform = n_filters - len(bands)
            uniform_centers = torch.linspace(
                self.wn_min + 100.0, 
                self.wn_max - 100.0, 
                n_uniform
            )
            centers = torch.cat([bio_centers, uniform_centers])
        
        # Add small random perturbation for diversity
        centers = centers + torch.randn(n_filters) * 10.0
        
        return centers
    
    def _clamp_cutoffs(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Enforce [wn_min, wn_max] range and minimum bandwidth of 10 cm^-1"""
        flo = torch.clamp(
            torch.min(self.f_low, self.f_high - 10.0), 
            self.wn_min, 
            self.wn_max - 10.0
        )
        fhi = torch.clamp(
            torch.max(self.f_high, self.f_low + 10.0), 
            self.wn_min + 10.0, 
            self.wn_max
        )
        return flo, fhi
    
    def _wn2omega(self, f_cm1: torch.Tensor, L: int) -> torch.Tensor:
        """
        Convert wavenumber to digital frequency based on actual sequence length.
        
        Args:
            f_cm1: Frequency in cm^-1
            L: Sequence length (number of spectral channels)
        
        Returns:
            Digital frequency in radians/sample
        """
        # Infer grid step from current input length
        delta = (self.wn_max - self.wn_min) / (L - 1)
        # Convert cm^-1 to sample index
        idx = (f_cm1 - self.wn_min) / delta
        # Convert to digital rad/sample
        return 2 * torch.pi * (idx / L)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, 1, C) where C is spectral length
        
        Returns:
            Tuple of:
                - Filtered output of shape (B, F, C) where F is out_channels
                - Tuple of (f_low_cm1, f_high_cm1) cutoff frequencies in cm^-1
        """
        B, _, C = x.shape
        flo, fhi = self._clamp_cutoffs()
        
        # Convert to digital angular frequency based on actual sequence length
        w1 = self._wn2omega(flo, C)[:, None]  # (F, 1)
        w2 = self._wn2omega(fhi, C)[:, None]  # (F, 1)
        
        n = self.n[None, :].to(x.device)  # (1, K)
        
        # Ideal band-pass: h[n] = (sin(w2*n) - sin(w1*n)) / (π*n)
        # Handle n=0 separately using limit
        num_hi = torch.where(n == 0, w2, torch.sin(w2 * n))
        num_lo = torch.where(n == 0, w1, torch.sin(w1 * n))
        den = torch.where(n == 0, torch.ones_like(n), n)
        h = (num_hi - num_lo) / (torch.pi * den)  # (F, K)
        
        # Apply cached Hamming window for stability
        h = h * self.win[None, :]
        
        # L2 normalize each filter to unit energy
        h = h / (torch.norm(h, dim=1, keepdim=True) + 1e-8)
        
        # Reshape for conv1d
        h = h[:, None, :]  # (F, 1, K)
        
        # Apply convolution
        y = F.conv1d(x, h, padding=self.ks//2)  # (B, F, C)
        
        return y, (flo, fhi)


class SpectralMambaBlock(nn.Module):
    """Mamba SSM for spectral sequence modeling"""
    
    def __init__(self,
                 d_model: int = 32,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 n_layers: int = 2):
        """
        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution width in Mamba
            expand: Expansion factor for inner dimension
            n_layers: Number of Mamba layers
        """
        super().__init__()
        
        self.layers = nn.ModuleList([
            Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand
            ) for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, L, D) where L is sequence length, D is d_model
        
        Returns:
            Output of shape (B, L, D)
        """
        for layer in self.layers:
            # Residual connection
            residual = x
            x = layer(x)
            x = self.dropout(x)
            x = x + residual
            x = self.norm(x)
        
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling over spectral dimension"""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape (B, L, D) where L is sequence length
        
        Returns:
            Pooled output of shape (B, D)
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (B, L, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted sum
        out = torch.sum(x * attn_weights, dim=1)  # (B, D)
        
        return out


class UNetBlock(nn.Module):
    """Basic U-Net encoder/decoder block"""
    
    def __init__(self, in_channels: int, out_channels: int, 
                 is_encoder: bool = True, use_bn: bool = True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        
        self.is_encoder = is_encoder
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class SSMUNet(nn.Module):
    """Complete SSMU-Net architecture"""
    
    def __init__(self, cfg: Dict):
        super().__init__()
        
        # Extract config
        sinc_cfg = cfg['model']['sinc']
        ssm_cfg = cfg['model']['ssm']
        unet_cfg = cfg['model']['unet']
        
        num_classes = cfg['model']['classes']
        embed_dim = cfg['model']['embed']
        self.chunk_size = cfg['model'].get('chunk_size', 2048)  # For memory efficiency

        # 1. Sinc spectral front-end
        self.sinc = SincConv1d(
            out_channels=sinc_cfg['filters'],
            kernel_size=sinc_cfg['kernel_size'],
            wn_min=sinc_cfg['wn_min'],
            wn_max=sinc_cfg['wn_max']
        )
        
        # 2. Mamba SSM for spectral modeling
        self.ssm = SpectralMambaBlock(
            d_model=sinc_cfg['filters'],
            d_state=ssm_cfg['d_state'],
            d_conv=ssm_cfg['d_conv'],
            expand=ssm_cfg['expand'],
            n_layers=ssm_cfg['layers']
        )
        
        # 3. Attention pooling
        self.pool = AttentionPooling(sinc_cfg['filters'])
        
        # 4. Projection to spatial feature dimension
        self.project = nn.Linear(sinc_cfg['filters'], embed_dim)
        
        # 5. U-Net for spatial segmentation
        base = unet_cfg['base']
        
        # Encoder
        self.enc1 = UNetBlock(embed_dim, base)
        self.enc2 = UNetBlock(base, base * 2)
        self.enc3 = UNetBlock(base * 2, base * 4)
        self.enc4 = UNetBlock(base * 4, base * 8)
        
        # Bottleneck
        self.bottleneck = UNetBlock(base * 8, base * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = UNetBlock(base * 16, base * 8, is_encoder=False)
        
        self.upconv3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = UNetBlock(base * 8, base * 4, is_encoder=False)
        
        self.upconv2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = UNetBlock(base * 4, base * 2, is_encoder=False)
        
        self.upconv1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = UNetBlock(base * 2, base, is_encoder=False)
        
        # Output
        self.out_conv = nn.Conv2d(base, num_classes, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _spectral_stack(self, x: torch.Tensor, chunk_size: int = 8192) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process spectral features in chunks to avoid OOM.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            chunk_size: Number of pixels to process at once
        
        Returns:
            Tuple of processed features and cutoff frequencies
        """
        B, C, H, W = x.shape
        z_all = []
        f_l_all, f_h_all = None, None
        
        # Flatten spatial dimensions
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)  # (BHW, 1, C)
        
        # Process in chunks
        for s in range(0, x_flat.size(0), chunk_size):
            xf = x_flat[s:s+chunk_size]
            
            # Sinc filtering
            z, (flo, fhi) = self.sinc(xf)  # (N, F, C)
            
            # Mamba SSM
            z = z.permute(0, 2, 1)  # (N, C, F)
            z = self.ssm(z)  # (N, C, F)
            
            # Attention pooling
            z = self.pool(z)  # (N, F)
            
            z_all.append(z)
            
            # Cutoffs are the same for all chunks, keep last reference
            f_l_all, f_h_all = flo, fhi
        
        # Concatenate all chunks
        z = torch.cat(z_all, dim=0)  # (BHW, F)
        
        return z, (f_l_all, f_h_all)
    
    def _pad_to_16(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Pad tensor to make H and W divisible by 16"""
        B, E, H, W = x.shape
        pad_h = (16 - H % 16) % 16
        pad_w = (16 - W % 16) % 16
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))  # pad right and bottom
        return x, H, W
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (B, C, H, W) where C is number of spectral channels
        
        Returns:
            Tuple of:
                - Segmentation logits of shape (B, num_classes, H, W)
                - Tuple of (f_low_cm1, f_high_cm1) cutoff frequencies with gradients
        """
        B, C, H, W = x.shape
        
        # Process spectral features in chunks
        z, (flo, fhi) = self._spectral_stack(x, self.chunk_size)
        
        # Project to embedding dimension
        z = self.project(z)  # (BHW, E)
        
        # Reshape back to spatial grid
        feat = z.reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, E, H, W)
        
        # Pad to multiple of 16 for U-Net
        feat, H0, W0 = self._pad_to_16(feat)
        
        # U-Net processing
        # Encoder with skip connections
        e1 = self.enc1(feat)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.upconv4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.upconv3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.upconv2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.upconv1(d2), e1], 1))
        
        # Output
        logits = self.out_conv(d1)
        
        # Crop back to original size
        logits = logits[:, :, :H0, :W0]
        
        # Return with gradients enabled for sparsity/overlap losses
        return logits, (flo, fhi)
    
    def get_cutoffs_cm1(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get learned Sinc filter cutoffs in cm^-1 for analysis (detached)"""
        with torch.no_grad():
            flo, fhi = self.sinc._clamp_cutoffs()
        return flo.detach().cpu(), fhi.detach().cpu()


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model(cfg: Dict) -> nn.Module:
    """Factory function to create model"""
    model = SSMUNet(cfg)
    
    # Print model statistics
    n_params = count_parameters(model)
    print(f"Created SSMU-Net with {n_params:,} trainable parameters")
    
    return model


if __name__ == "__main__":
    # Test model creation
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "ssmu_net" / "config.yaml"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    model = create_model(cfg)
    
    # Test forward pass
    batch_size = 2
    n_channels = 451  # Number of spectral channels
    height, width = 128, 128
    
    x = torch.randn(batch_size, n_channels, height, width)
    logits, (f_low, f_high) = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Expected: ({batch_size}, {cfg['model']['classes']}, {height}, {width})")
    print(f"Filter cutoffs: f_low shape={f_low.shape}, f_high shape={f_high.shape}")
    print(f"Sample cutoffs (cm^-1): [{f_low[0]:.1f}, {f_high[0]:.1f}]")