import torch
import torch.nn as nn

class VanillaGANDiscriminator(nn.Module):
    """
    Vanilla GAN Discriminator using fully connected layers
    """
    def __init__(self, img_shape=784):
        super(VanillaGANDiscriminator, self).__init__()
        
        self.model = nn.Sequential(
            # First layer
            nn.Linear(img_shape, 1024),
            nn.LeakyReLU(0.25),
            
            # Second layer
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.25),
            
            # Third layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.25),
            
            # Output layer
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        """
        Forward pass through the discriminator
        Args:
            img: flattened image of shape (batch_size, img_shape)
        Returns:
            Probability that the image is real (batch_size, 1)
        """
        return self.model(img)
