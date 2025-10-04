import torch
import torch.nn as nn

class VanillaGANGenerator(nn.Module):
    """
    Vanilla GAN Generator using fully connected layers
    """
    def __init__(self, noise_dim=50, img_shape=784):
        super(VanillaGANGenerator, self).__init__()
        self.img_shape = img_shape
        
        self.model = nn.Sequential(
            # First upsampling layer
            nn.Linear(noise_dim, 128, bias=False),
            nn.BatchNorm1d(128, 0.8),
            nn.LeakyReLU(0.25),
            
            # Second upsampling layer
            nn.Linear(128, 256, bias=False),
            nn.BatchNorm1d(256, 0.8),
            nn.LeakyReLU(0.25),
            
            # Third upsampling layer
            nn.Linear(256, 512, bias=False),
            nn.BatchNorm1d(512, 0.8),
            nn.LeakyReLU(0.25),
            
            # Final upsampling layer
            nn.Linear(512, img_shape),
            nn.Tanh()
        )

    def forward(self, z):
        """
        Forward pass through the generator
        Args:
            z: noise vector of shape (batch_size, noise_dim)
        Returns:
            Generated images of shape (batch_size, img_shape)
        """
        return self.model(z)
