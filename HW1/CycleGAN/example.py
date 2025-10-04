"""
Example usage of CycleGAN implementation
This script demonstrates how to set up and use the CycleGAN for training and inference
"""

import torch
import os
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from CycleGan_Generator import Generator
from CycleGan_Discriminator import Discriminator
from dataset import CycleGANDataset, get_transforms


def demonstrate_architecture():
    """Demonstrate the CycleGAN architecture"""
    print("CycleGAN Architecture Demonstration")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create models
    gen_H = Generator(img_channels=3, num_residuals=9).to(device)
    gen_Z = Generator(img_channels=3, num_residuals=9).to(device) 
    disc_H = Discriminator(in_channels=3).to(device)
    disc_Z = Discriminator(in_channels=3).to(device)
    
    # Create sample input
    batch_size = 1
    sample_x = torch.randn(batch_size, 3, 256, 256).to(device)  # Domain X (e.g., horses)
    sample_y = torch.randn(batch_size, 3, 256, 256).to(device)  # Domain Y (e.g., zebras)
    
    print(f"\nInput shapes:")
    print(f"Domain X: {sample_x.shape}")
    print(f"Domain Y: {sample_y.shape}")
    
    # Forward pass through generators
    with torch.no_grad():
        fake_y = gen_H(sample_x)  # X -> Y (horse -> zebra)
        fake_x = gen_Z(sample_y)  # Y -> X (zebra -> horse)
        
        # Cycle consistency
        cycle_x = gen_Z(fake_y)   # X -> Y -> X
        cycle_y = gen_H(fake_x)   # Y -> X -> Y
        
        # Discriminator outputs
        disc_real_x = disc_H(sample_x)
        disc_fake_x = disc_H(fake_x)
        disc_real_y = disc_Z(sample_y)
        disc_fake_y = disc_Z(fake_y)
    
    print(f"\nGenerator outputs:")
    print(f"G_H(X): {fake_y.shape} (X->Y translation)")
    print(f"G_Z(Y): {fake_x.shape} (Y->X translation)")
    
    print(f"\nCycle consistency:")
    print(f"X -> Y -> X: {cycle_x.shape}")
    print(f"Y -> X -> Y: {cycle_y.shape}")
    
    print(f"\nDiscriminator outputs (patch predictions):")
    print(f"D_H(real_X): {disc_real_x.shape}")
    print(f"D_H(fake_X): {disc_fake_x.shape}")
    print(f"D_Z(real_Y): {disc_real_y.shape}")
    print(f"D_Z(fake_Y): {disc_fake_y.shape}")
    
    # Model parameter counts
    print(f"\nModel Parameters:")
    print(f"Generator H: {sum(p.numel() for p in gen_H.parameters()):,}")
    print(f"Generator Z: {sum(p.numel() for p in gen_Z.parameters()):,}")
    print(f"Discriminator H: {sum(p.numel() for p in disc_H.parameters()):,}")
    print(f"Discriminator Z: {sum(p.numel() for p in disc_Z.parameters()):,}")
    
    total_params = (sum(p.numel() for p in gen_H.parameters()) + 
                   sum(p.numel() for p in gen_Z.parameters()) + 
                   sum(p.numel() for p in disc_H.parameters()) + 
                   sum(p.numel() for p in disc_Z.parameters()))
    print(f"Total Parameters: {total_params:,}")


def create_sample_config():
    """Create a sample configuration for training"""
    config = {
        # Training parameters
        'num_epochs': 200,
        'learning_rate': 2e-4,
        'batch_size': 1,
        'num_workers': 4,
        'lambda_cycle': 10,
        'lambda_identity': 0.5,
        'num_residuals': 9,
        'image_size': 256,
        
        # Data directories
        'train_dir_a': "data/trainA",
        'train_dir_b': "data/trainB",
        'val_dir_a': "data/testA",
        'val_dir_b': "data/testB",
        
        # Output directories
        'checkpoint_dir': "checkpoints",
        'sample_dir': "samples",
        'checkpoint_gen_h': "checkpoints/gen_h.pth.tar",
        'checkpoint_gen_z': "checkpoints/gen_z.pth.tar",
        'checkpoint_disc_h': "checkpoints/disc_h.pth.tar",
        'checkpoint_disc_z': "checkpoints/disc_z.pth.tar",
        
        # Training settings
        'load_model': False,
        'save_model': True,
    }
    return config


def training_example():
    """Example of how to start training"""
    print("\nTraining Example")
    print("=" * 30)
    
    config = create_sample_config()
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"\nTo start training:")
    print(f"1. Prepare your dataset in the following structure:")
    print(f"   data/")
    print(f"   ├── trainA/  (Domain A training images)")
    print(f"   ├── trainB/  (Domain B training images)")
    print(f"   ├── testA/   (Domain A test images)")
    print(f"   └── testB/   (Domain B test images)")
    print(f"")
    print(f"2. Run: python train.py")
    print(f"")
    print(f"3. Monitor training progress:")
    print(f"   - Checkpoints saved in: {config['checkpoint_dir']}/")
    print(f"   - Sample outputs in: {config['sample_dir']}/")


def loss_function_example():
    """Demonstrate the loss functions used in CycleGAN"""
    print("\nCycleGAN Loss Functions")
    print("=" * 35)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create sample data
    real_x = torch.randn(1, 3, 256, 256).to(device)
    real_y = torch.randn(1, 3, 256, 256).to(device)
    
    # Create models
    gen_H = Generator().to(device)
    gen_Z = Generator().to(device)
    disc_H = Discriminator().to(device)
    disc_Z = Discriminator().to(device)
    
    # Loss functions
    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    
    with torch.no_grad():
        # Forward translations
        fake_y = gen_H(real_x)  # X -> Y
        fake_x = gen_Z(real_y)  # Y -> X
        
        # Cycle translations
        cycle_x = gen_Z(fake_y)  # X -> Y -> X
        cycle_y = gen_H(fake_x)  # Y -> X -> Y
        
        # Discriminator outputs
        disc_real_y = disc_Z(real_y)
        disc_fake_y = disc_Z(fake_y)
        disc_real_x = disc_H(real_x)
        disc_fake_x = disc_H(fake_x)
        
        # 1. Adversarial Loss
        adv_loss_gen_h = mse_loss(disc_fake_y, torch.ones_like(disc_fake_y))
        adv_loss_gen_z = mse_loss(disc_fake_x, torch.ones_like(disc_fake_x))
        
        # 2. Cycle Consistency Loss
        cycle_loss_x = l1_loss(real_x, cycle_x)
        cycle_loss_y = l1_loss(real_y, cycle_y)
        
        # 3. Identity Loss (optional)
        identity_x = gen_Z(real_x)
        identity_y = gen_H(real_y)
        identity_loss_x = l1_loss(real_x, identity_x)
        identity_loss_y = l1_loss(real_y, identity_y)
        
        # Total Generator Loss
        lambda_cycle = 10
        lambda_identity = 0.5
        
        total_gen_loss = (adv_loss_gen_h + adv_loss_gen_z + 
                         lambda_cycle * (cycle_loss_x + cycle_loss_y) +
                         lambda_identity * (identity_loss_x + identity_loss_y))
        
        # Discriminator Loss
        disc_loss_real_x = mse_loss(disc_real_x, torch.ones_like(disc_real_x))
        disc_loss_fake_x = mse_loss(disc_fake_x, torch.zeros_like(disc_fake_x))
        disc_loss_x = (disc_loss_real_x + disc_loss_fake_x) / 2
        
        disc_loss_real_y = mse_loss(disc_real_y, torch.ones_like(disc_real_y))
        disc_loss_fake_y = mse_loss(disc_fake_y, torch.zeros_like(disc_fake_y))
        disc_loss_y = (disc_loss_real_y + disc_loss_fake_y) / 2
        
        total_disc_loss = (disc_loss_x + disc_loss_y) / 2
    
    print(f"Loss Components:")
    print(f"1. Adversarial Losses:")
    print(f"   Generator H (X->Y): {adv_loss_gen_h.item():.4f}")
    print(f"   Generator Z (Y->X): {adv_loss_gen_z.item():.4f}")
    
    print(f"2. Cycle Consistency Losses:")
    print(f"   Cycle X (X->Y->X): {cycle_loss_x.item():.4f}")
    print(f"   Cycle Y (Y->X->Y): {cycle_loss_y.item():.4f}")
    
    print(f"3. Identity Losses:")
    print(f"   Identity X: {identity_loss_x.item():.4f}")
    print(f"   Identity Y: {identity_loss_y.item():.4f}")
    
    print(f"\nTotal Losses:")
    print(f"Generator Loss: {total_gen_loss.item():.4f}")
    print(f"Discriminator Loss: {total_disc_loss.item():.4f}")
    
    print(f"\nLoss Weights:")
    print(f"λ_cycle = {lambda_cycle}")
    print(f"λ_identity = {lambda_identity}")


def main():
    """Main demonstration function"""
    print("CycleGAN Implementation Example")
    print("=" * 60)
    print()
    
    # 1. Architecture demonstration
    demonstrate_architecture()
    
    # 2. Loss function example
    loss_function_example()
    
    # 3. Training example
    training_example()
    
    print("\n" + "=" * 60)
    print("Example completed! Check the README.md for detailed usage instructions.")


if __name__ == "__main__":
    main()
