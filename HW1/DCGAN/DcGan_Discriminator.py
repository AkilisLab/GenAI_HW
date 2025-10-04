import torch
import torch.nn as nn

class DCGANDiscriminator(nn.Module):
    def __init__(self, d_channels=64, img_channels=3):
        super(DCGANDiscriminator, self).__init__()
        
        self.net = nn.Sequential(
            # Input: (img_channels, 64, 64)
            
            # Conv1: (64 x 32 x 32)
            nn.Conv2d(img_channels, d_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv2: (128 x 16 x 16)
            nn.Conv2d(d_channels, d_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv3: (256 x 8 x 8)
            nn.Conv2d(d_channels * 2, d_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Conv4: (512 x 4 x 4)
            nn.Conv2d(d_channels * 4, d_channels * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(d_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final Conv: output (1 x 1 x 1)
            nn.Conv2d(d_channels * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
        # return self.net(x).view(-1, 1).squeeze(1)


# Example usage
if __name__ == "__main__":
    D = DCGANDiscriminator(d_channels=64, img_channels=3)
    x = torch.randn(16, 3, 64, 64)  # batch of 16 RGB images
    out = D(x)
    print(out.shape)  # (16,) each value ∈ [0,1]
