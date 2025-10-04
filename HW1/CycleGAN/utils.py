"""
Utility functions for CycleGAN training and inference
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from torchvision import utils as vutils


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init(m):
    """
    Custom weights initialization for neural networks
    Apply normal initialization to Conv2d and InstanceNorm2d layers
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # Handle both Conv2d and ConvTranspose2d
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('InstanceNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


def create_output_directories(base_dir):
    """
    Create output directories for checkpoints and samples
    
    Args:
        base_dir: Base directory path
        
    Returns:
        Tuple of (checkpoint_dir, sample_dir)
    """
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    sample_dir = os.path.join(base_dir, "samples")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    return checkpoint_dir, sample_dir


def save_checkpoint(model, optimizer, epoch, filename):
    """
    Save model checkpoint with additional information
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch: Current epoch number
        filename: Checkpoint filename
    """
    print(f"=> Saving checkpoint: {filename}")
    checkpoint = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer=None, device=None):
    """
    Load model checkpoint
    
    Args:
        checkpoint_file: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        device: Device to map the checkpoint to
        
    Returns:
        Loaded epoch number
    """
    print(f"=> Loading checkpoint: {checkpoint_file}")
    
    if device:
        checkpoint = torch.load(checkpoint_file, map_location=device)
    else:
        checkpoint = torch.load(checkpoint_file)
    
    model.load_state_dict(checkpoint["state_dict"])
    
    if optimizer and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    epoch = checkpoint.get("epoch", 0)
    return epoch


def save_sample_images(gen_H, gen_Z, dataloader, epoch, sample_dir, device, num_samples=4):
    """
    Generate and save sample translations
    
    Args:
        gen_H: Generator H (Domain A -> Domain B)
        gen_Z: Generator Z (Domain B -> Domain A)
        dataloader: Validation dataloader
        epoch: Current epoch
        sample_dir: Directory to save samples
        device: Device to run inference on
        num_samples: Number of sample pairs to generate
    """
    gen_H.eval()
    gen_Z.eval()
    
    try:
        # Get a batch of data
        real_A, real_B = next(iter(dataloader))
        real_A = real_A[:num_samples].to(device)
        real_B = real_B[:num_samples].to(device)
        
        with torch.no_grad():
            # Generate translations
            fake_B = gen_H(real_A)  # A -> B
            fake_A = gen_Z(real_B)  # B -> A
            
            # Cycle translations
            cycle_A = gen_Z(fake_B)  # A -> B -> A
            cycle_B = gen_H(fake_A)  # B -> A -> B
            
            # Denormalize images for saving
            real_A = real_A * 0.5 + 0.5
            real_B = real_B * 0.5 + 0.5
            fake_A = fake_A * 0.5 + 0.5
            fake_B = fake_B * 0.5 + 0.5
            cycle_A = cycle_A * 0.5 + 0.5
            cycle_B = cycle_B * 0.5 + 0.5
            
            # Create comparison grid
            comparison = torch.cat([
                real_A, fake_B, cycle_A,  # A -> B -> A
                real_B, fake_A, cycle_B   # B -> A -> B
            ], dim=0)
            
            # Save the comparison
            grid = vutils.make_grid(comparison, nrow=3, padding=2, normalize=False)
            vutils.save_image(grid, os.path.join(sample_dir, f"epoch_{epoch:03d}_comparison.png"))
            
            # Save individual domain translations
            vutils.save_image(fake_B, os.path.join(sample_dir, f"epoch_{epoch:03d}_A_to_B.png"),
                             nrow=2, padding=2, normalize=False)
            vutils.save_image(fake_A, os.path.join(sample_dir, f"epoch_{epoch:03d}_B_to_A.png"),
                             nrow=2, padding=2, normalize=False)
            
    except Exception as e:
        print(f"Warning: Could not save sample images - {e}")
    
    finally:
        gen_H.train()
        gen_Z.train()


def plot_training_losses(g_losses_H, g_losses_Z, d_losses_H, d_losses_Z, cycle_losses, 
                        identity_losses, save_path):
    """
    Plot and save training loss curves
    
    Args:
        g_losses_H: Generator H losses
        g_losses_Z: Generator Z losses
        d_losses_H: Discriminator H losses
        d_losses_Z: Discriminator Z losses
        cycle_losses: Cycle consistency losses
        identity_losses: Identity losses
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('CycleGAN Training Losses', fontsize=16)
    
    # Generator losses
    axes[0, 0].plot(g_losses_H, label='Generator H (A->B)', alpha=0.8)
    axes[0, 0].plot(g_losses_Z, label='Generator Z (B->A)', alpha=0.8)
    axes[0, 0].set_title('Generator Losses')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator losses
    axes[0, 1].plot(d_losses_H, label='Discriminator H', alpha=0.8)
    axes[0, 1].plot(d_losses_Z, label='Discriminator Z', alpha=0.8)
    axes[0, 1].set_title('Discriminator Losses')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cycle loss
    axes[1, 0].plot(cycle_losses, label='Cycle Consistency', color='green', alpha=0.8)
    axes[1, 0].set_title('Cycle Consistency Loss')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Identity loss
    if identity_losses and len(identity_losses) > 0:
        axes[1, 1].plot(identity_losses, label='Identity', color='orange', alpha=0.8)
        axes[1, 1].set_title('Identity Loss')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Identity Loss\nNot Used', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].set_title('Identity Loss')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training losses plot saved to: {save_path}")


def print_training_stats(epoch, batch_idx, total_batches, losses_dict, time_elapsed=None):
    """
    Print training statistics in a formatted way
    
    Args:
        epoch: Current epoch
        batch_idx: Current batch index
        total_batches: Total number of batches
        losses_dict: Dictionary containing loss values
        time_elapsed: Elapsed time (optional)
    """
    progress = (batch_idx + 1) / total_batches * 100
    
    stats = f"Epoch [{epoch:3d}] [{batch_idx+1:4d}/{total_batches:4d}] ({progress:5.1f}%)"
    
    for loss_name, loss_value in losses_dict.items():
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()
        stats += f" | {loss_name}: {loss_value:.4f}"
    
    if time_elapsed:
        stats += f" | Time: {time_elapsed:.2f}s"
    
    print(stats)


def denormalize_image(tensor):
    """
    Denormalize image tensor from [-1, 1] to [0, 1]
    
    Args:
        tensor: Image tensor with values in [-1, 1]
        
    Returns:
        Denormalized tensor with values in [0, 1]
    """
    return tensor * 0.5 + 0.5


# Legacy functions for backward compatibility
def save_some_examples(gen_H, gen_Z, val_loader, epoch, folder, device):
    """Legacy function - redirects to save_sample_images"""
    save_sample_images(gen_H, gen_Z, val_loader, epoch, folder, device)


def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


if __name__ == "__main__":
    print("CycleGAN Utils module loaded successfully!")
