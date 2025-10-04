import torch
import torch.nn as nn


class ConvolutionalBlock(nn.Module):
    """
    Convolutional block for CycleGAN Generator
    Can be used for both downsampling and upsampling
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        is_downsampling: bool = True,
        add_activation: bool = True,
        **kwargs
    ):
        super().__init__()
        if is_downsampling:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )
        else:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True) if add_activation else nn.Identity(),
            )

    def forward(self, x):
        return self.conv(x)


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers
    Used in the middle of the generator
    """
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(
                channels, channels, 
                add_activation=True, 
                kernel_size=3, 
                padding=1
            ),
            ConvolutionalBlock(
                channels, channels, 
                add_activation=False, 
                kernel_size=3, 
                padding=1
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN Generator
    Architecture: c7s1-64, d128, d256, R256×6/9, u128, u64, c7s1-3
    
    - c7s1-k: 7×7 Convolution-InstanceNorm-ReLU layer with k filters and stride 1
    - dk: 3×3 Convolution-InstanceNorm-ReLU layer with k filters and stride 2  
    - Rk: residual block with k filters
    - uk: 3×3 fractional-strided-Convolution-InstanceNorm-ReLU layer with k filters and stride 1/2
    """
    def __init__(
        self, 
        img_channels: int = 3, 
        num_features: int = 64, 
        num_residuals: int = 6
    ):
        super().__init__()
        
        # Initial layer: c7s1-64
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        
        # Downsampling layers: d128, d256
        self.downsampling_layers = nn.ModuleList([
            ConvolutionalBlock(
                num_features,
                num_features * 2,
                is_downsampling=True,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            ConvolutionalBlock(
                num_features * 2,
                num_features * 4,
                is_downsampling=True,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
        ])
        
        # Residual layers: R256×6 or R256×9
        self.residual_layers = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )
        
        # Upsampling layers: u128, u64
        self.upsampling_layers = nn.ModuleList([
            ConvolutionalBlock(
                num_features * 4,
                num_features * 2,
                is_downsampling=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            ConvolutionalBlock(
                num_features * 2,
                num_features,
                is_downsampling=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
        ])
        
        # Final layer: c7s1-3
        self.final_layer = nn.Conv2d(
            num_features,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = self.initial_layer(x)
        
        for layer in self.downsampling_layers:
            x = layer(x)
        
        x = self.residual_layers(x)
        
        for layer in self.upsampling_layers:
            x = layer(x)
        
        return torch.tanh(self.final_layer(x))


def test_generator():
    """Test the generator with a sample input"""
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, num_residuals=9)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {gen(x).shape}")
    print(f"Number of parameters: {sum(p.numel() for p in gen.parameters())}")


if __name__ == "__main__":
    test_generator()