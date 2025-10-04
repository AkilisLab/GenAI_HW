import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, z_dim=100, g_channels=64, img_channels=3):
        super(DCGANGenerator, self).__init__()
        # g_channels is the base channel size; paper used 64
        # so first layer will be 1024 = g_channels * 16
        
        self.net = nn.Sequential(
            # Input Z: (batch, z_dim, 1, 1)
            
            # First: project and reshape → 1024 x 4 x 4
            nn.ConvTranspose2d(z_dim, g_channels * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(g_channels * 16),
            nn.ReLU(True),
            
            # 512 x 8 x 8
            nn.ConvTranspose2d(g_channels * 16, g_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_channels * 8),
            nn.ReLU(True),
            
            # 256 x 16 x 16
            nn.ConvTranspose2d(g_channels * 8, g_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_channels * 4),
            nn.ReLU(True),
            
            # 128 x 32 x 32
            nn.ConvTranspose2d(g_channels * 4, g_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(g_channels * 2),
            nn.ReLU(True),
            
            # Final layer: 3 x 64 x 64 (RGB image)
            nn.ConvTranspose2d(g_channels * 2, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # output range [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


# Example usage
if __name__ == "__main__":
    z_dim = 100
    G = DCGANGenerator(z_dim=z_dim, g_channels=64, img_channels=3)
    z = torch.randn(16, z_dim, 1, 1)  # batch of 16 latent vectors
    fake_images = G(z)
    print(fake_images.shape)  # (16, 3, 64, 64)
