import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import utils as vutils

def weights_init(m):
    """
    Custom weights initialization for neural networks
    """
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)

def generate_noise(batch_size, noise_dim, device):
    """
    Generate random noise for the generator
    """
    return torch.randn(batch_size, noise_dim, device=device)

def save_sample_images(generator, fixed_noise, epoch, batch_idx, output_dir):
    """
    Generate and save sample images from the generator
    """
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise)
        # Reshape to image format (batch_size, 1, 28, 28)
        fake_images = fake_images.view(-1, 1, 28, 28)
        
        # Create grid of images
        grid = vutils.make_grid(fake_images[:16], nrow=4, padding=2, normalize=True)
        
        # Save the grid
        vutils.save_image(grid, f"{output_dir}/fake_samples_epoch_{epoch:03d}_batch_{batch_idx:03d}.png")
    
    generator.train()

def plot_training_progress(g_losses, d_losses, output_dir):
    """
    Plot and save training loss curves
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="Generator Loss")
    plt.plot(d_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/training_losses.png", dpi=150, bbox_inches='tight')
    plt.show()

def plot_real_vs_fake_comparison(real_batch, fake_batch, output_dir):
    """
    Plot comparison between real and fake images
    """
    plt.figure(figsize=(15, 8))
    
    # Plot real images
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    real_grid = vutils.make_grid(real_batch[:16], nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(real_grid.cpu(), (1, 2, 0)), cmap='gray')
    
    # Plot fake images
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Generated Images")
    fake_grid = vutils.make_grid(fake_batch[:16], nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(fake_grid.cpu(), (1, 2, 0)), cmap='gray')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/real_vs_fake_comparison.png", dpi=150, bbox_inches='tight')
    plt.show()

def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, 
                   epoch, g_loss, d_loss, checkpoint_dir):
    """
    Save model checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'g_loss': g_loss,
        'd_loss': d_loss,
    }
    torch.save(checkpoint, f"{checkpoint_dir}/checkpoint_epoch_{epoch:03d}.pth")

def create_output_directories(base_dir):
    """
    Create necessary output directories
    """
    output_dir = f"{base_dir}/outputs"
    checkpoint_dir = f"{base_dir}/checkpoints"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return output_dir, checkpoint_dir

def print_training_stats(epoch, total_epochs, batch_idx, total_batches, 
                        g_loss, d_loss, d_real_acc, d_fake_acc):
    """
    Print training statistics
    """
    print(f'[{epoch}/{total_epochs}][{batch_idx}/{total_batches}] '
          f'Loss_D: {d_loss:.4f} Loss_G: {g_loss:.4f} '
          f'D(x): {d_real_acc:.4f} D(G(z)): {d_fake_acc:.4f}')

def plot_loss_progression(g_losses, d_losses, window_size=100):
    """
    Plot smoothed loss progression during training
    """
    def moving_average(data, window_size):
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    if len(g_losses) > window_size:
        g_smooth = moving_average(g_losses, window_size)
        d_smooth = moving_average(d_losses, window_size)
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(g_losses, alpha=0.3, color='blue', label='Generator (raw)')
        plt.plot(range(window_size-1, len(g_losses)), g_smooth, color='blue', label=f'Generator (MA-{window_size})')
        plt.plot(d_losses, alpha=0.3, color='red', label='Discriminator (raw)')
        plt.plot(range(window_size-1, len(d_losses)), d_smooth, color='red', label=f'Discriminator (MA-{window_size})')
        plt.title('Training Losses (Smoothed)')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(g_losses[-1000:], alpha=0.7, color='blue', label='Generator (last 1000)')
        plt.plot(d_losses[-1000:], alpha=0.7, color='red', label='Discriminator (last 1000)')
        plt.title('Recent Training Losses')
        plt.xlabel('Iteration (last 1000)')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
