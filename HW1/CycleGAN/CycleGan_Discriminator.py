import torch
import torch.nn as nn


class ConvInstanceNormLeakyReLUBlock(nn.Module):
    """
    Convolutional block with InstanceNorm and LeakyReLU for Discriminator
    Ck denotes a 4×4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2
    """
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    """
    CycleGAN Discriminator (PatchGAN)
    Architecture: C64-C128-C256-C512
    
    - Ck: 4×4 Convolution-InstanceNorm-LeakyReLU layer with k filters and stride 2
    - Uses LeakyReLU with slope 0.2
    - Final layer produces 1-dimensional output (patch classification)
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Initial layer: C64 (no InstanceNorm for first layer)
        self.initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        
        # Subsequent layers: C128, C256, C512
        for feature in features[1:]:
            layers.append(
                ConvInstanceNormLeakyReLUBlock(
                    in_channels,
                    feature,
                    stride=1 if feature == features[-1] else 2,  # Last layer has stride 1
                )
            )
            in_channels = feature

        # Final convolution to produce 1-dimensional output
        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial_layer(x)
        return torch.sigmoid(self.model(x))


def test_discriminator():
    """Test the discriminator with a sample input"""
    img_channels = 3
    img_size = 256
    x = torch.randn((2, img_channels, img_size, img_size))
    disc = Discriminator(img_channels)
    print(f"Input shape: {x.shape}")
    output = disc(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in disc.parameters())}")


if __name__ == "__main__":
    test_discriminator()