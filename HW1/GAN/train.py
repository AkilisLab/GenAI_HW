import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
import os

# Import our custom modules
from VanillaGan_Generator import VanillaGANGenerator
from VanillaGan_Discriminator import VanillaGANDiscriminator
from dataset import prepare_mnist_dataset, get_sample_batch
from utils import (weights_init, generate_noise, save_sample_images, 
                   plot_training_progress, plot_real_vs_fake_comparison,
                   save_checkpoint, create_output_directories, 
                   print_training_stats, plot_loss_progression)

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Configuration
class Config:
    # Model hyperparameters
    noise_dim = 50          # Size of noise vector
    img_shape = 28 * 28     # Flattened image size (28x28)
    
    # Training hyperparameters
    num_epochs = 25         # Reduced for faster training
    batch_size = 128
    lr = 0.0002            # Learning rate
    beta1 = 0.5            # Beta1 for Adam optimizer
    
    # Training settings
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print_freq = 50        # Print statistics every N batches
    save_freq = 5          # Save samples every N epochs
    
    # Labels
    real_label = 1.0
    fake_label = 0.0

def train_vanilla_gan():
    """
    Main training function for Vanilla GAN
    """
    config = Config()
    set_seed(42)
    
    print(f"Using device: {config.device}")
    print(f"Training for {config.num_epochs} epochs with batch size {config.batch_size}")
    
    # Create output directories
    output_dir, checkpoint_dir = create_output_directories("./runs")
    
    # Prepare dataset
    print("Loading MNIST dataset...")
    dataloader = prepare_mnist_dataset(batch_size=config.batch_size)
    print(f"Dataset loaded. Total batches per epoch: {len(dataloader)}")
    
    # Initialize models
    print("Initializing models...")
    generator = VanillaGANGenerator(noise_dim=config.noise_dim, img_shape=config.img_shape).to(config.device)
    discriminator = VanillaGANDiscriminator(img_shape=config.img_shape).to(config.device)
    
    # Apply custom weights initialization
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Print model architectures
    print("\nGenerator Architecture:")
    print(generator)
    print("\nDiscriminator Architecture:")
    print(discriminator)
    
    # Loss function
    criterion = nn.BCELoss()
    
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, 0.999))
    
    # Fixed noise for consistent sample generation
    fixed_noise = generate_noise(16, config.noise_dim, config.device)
    
    # Training loop
    print("\nStarting Training...")
    print("-" * 50)
    
    g_losses = []
    d_losses = []
    
    for epoch in range(config.num_epochs):
        for batch_idx, (real_images,) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(config.device)
            
            # Flatten images for fully connected layers
            real_images = real_images.view(batch_size, -1)
            
            # Create labels
            real_labels = torch.full((batch_size, 1), config.real_label, device=config.device, dtype=torch.float)
            fake_labels = torch.full((batch_size, 1), config.fake_label, device=config.device, dtype=torch.float)
            
            #############################
            # Train Discriminator
            #############################
            discriminator.zero_grad()
            
            # Train with real images
            real_output = discriminator(real_images)
            d_loss_real = criterion(real_output, real_labels)
            d_loss_real.backward()
            d_real_acc = real_output.mean().item()
            
            # Train with fake images
            noise = generate_noise(batch_size, config.noise_dim, config.device)
            fake_images = generator(noise)
            fake_output = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_output, fake_labels)
            d_loss_fake.backward()
            d_fake_acc = fake_output.mean().item()
            
            # Update discriminator
            d_loss = d_loss_real + d_loss_fake
            optimizer_d.step()
            
            #############################
            # Train Generator
            #############################
            generator.zero_grad()
            
            # Train generator to fool discriminator
            fake_output = discriminator(fake_images)
            g_loss = criterion(fake_output, real_labels)  # We want fake to be classified as real
            g_loss.backward()
            optimizer_g.step()
            
            # Save losses for plotting
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())
            
            # Print statistics
            if batch_idx % config.print_freq == 0:
                print_training_stats(epoch, config.num_epochs, batch_idx, len(dataloader),
                                    g_loss.item(), d_loss.item(), d_real_acc, d_fake_acc)
        
        # Save sample images
        if epoch % config.save_freq == 0 or epoch == config.num_epochs - 1:
            save_sample_images(generator, fixed_noise, epoch, 0, output_dir)
            save_checkpoint(generator, discriminator, optimizer_g, optimizer_d,
                          epoch, g_loss.item(), d_loss.item(), checkpoint_dir)
        
        # Plot loss progression periodically
        if epoch > 0 and epoch % 10 == 0:
            plot_loss_progression(g_losses, d_losses)
    
    print("\nTraining completed!")
    print("-" * 50)
    
    # Final evaluation and plotting
    print("Generating final plots...")
    
    # Plot training losses
    plot_training_progress(g_losses, d_losses, output_dir)
    
    # Generate comparison plot
    real_batch = get_sample_batch(dataloader)
    real_images_viz = real_batch[0][:16]  # Take first 16 images
    
    generator.eval()
    with torch.no_grad():
        fake_images_viz = generator(fixed_noise)
        fake_images_viz = fake_images_viz.view(-1, 1, 28, 28)  # Reshape to image format
    
    plot_real_vs_fake_comparison(real_images_viz, fake_images_viz, output_dir)
    
    # Final sample generation
    save_sample_images(generator, fixed_noise, config.num_epochs, 0, output_dir)
    
    print(f"All outputs saved to: {output_dir}")
    print(f"Model checkpoints saved to: {checkpoint_dir}")

if __name__ == '__main__':
    train_vanilla_gan()