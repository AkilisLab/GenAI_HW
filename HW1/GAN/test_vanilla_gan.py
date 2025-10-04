import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils as vutils
import os

from VanillaGan_Generator import VanillaGANGenerator
from VanillaGan_Discriminator import VanillaGANDiscriminator
from dataset import prepare_mnist_dataset, get_sample_batch
from utils import generate_noise, create_output_directories

def load_checkpoint(checkpoint_path, generator, discriminator, device):
    """
    Load model checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    epoch = checkpoint['epoch']
    g_loss = checkpoint['g_loss']
    d_loss = checkpoint['d_loss']
    
    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Generator loss: {g_loss:.4f}, Discriminator loss: {d_loss:.4f}")
    
    return epoch

def generate_samples(generator, num_samples=64, noise_dim=50, device='cpu'):
    """
    Generate samples from the trained generator
    """
    generator.eval()
    with torch.no_grad():
        noise = generate_noise(num_samples, noise_dim, device)
        fake_images = generator(noise)
        # Reshape to image format (batch_size, 1, 28, 28)
        fake_images = fake_images.view(num_samples, 1, 28, 28)
    
    return fake_images

def interpolate_in_latent_space(generator, noise_dim=50, device='cpu', steps=10):
    """
    Perform interpolation in the latent space between two random points
    """
    generator.eval()
    
    # Generate two random points in latent space
    z1 = generate_noise(1, noise_dim, device)
    z2 = generate_noise(1, noise_dim, device)
    
    # Create interpolation steps
    alphas = torch.linspace(0, 1, steps, device=device).view(-1, 1)
    interpolated_z = alphas * z2 + (1 - alphas) * z1
    
    with torch.no_grad():
        interpolated_images = generator(interpolated_z)
        interpolated_images = interpolated_images.view(steps, 1, 28, 28)
    
    return interpolated_images

def evaluate_discriminator(discriminator, dataloader, device, num_batches=10):
    """
    Evaluate discriminator accuracy on real vs fake images
    """
    discriminator.eval()
    
    real_correct = 0
    fake_correct = 0
    total_real = 0
    total_fake = 0
    
    # Test on real images
    with torch.no_grad():
        for i, (real_images,) in enumerate(dataloader):
            if i >= num_batches:
                break
                
            real_images = real_images.to(device)
            real_images = real_images.view(real_images.size(0), -1)
            
            real_output = discriminator(real_images)
            real_predictions = (real_output > 0.5).float()
            real_correct += (real_predictions == 1).sum().item()
            total_real += real_images.size(0)
    
    print(f"Discriminator accuracy on real images: {100 * real_correct / total_real:.2f}%")
    return real_correct / total_real

def test_vanilla_gan():
    """
    Test the trained Vanilla GAN
    """
    # Configuration
    noise_dim = 50
    img_shape = 28 * 28
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Create output directory for test results
    test_output_dir = "./test_outputs"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # Initialize models
    generator = VanillaGANGenerator(noise_dim=noise_dim, img_shape=img_shape).to(device)
    discriminator = VanillaGANDiscriminator(img_shape=img_shape).to(device)
    
    # Find the latest checkpoint
    checkpoint_dir = "./runs/checkpoints"
    if not os.path.exists(checkpoint_dir):
        print("No checkpoints found! Please train the model first.")
        return
    
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        print("No checkpoint files found!")
        return
    
    # Load the latest checkpoint (highest epoch number)
    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Loading checkpoint: {latest_checkpoint}")
    epoch = load_checkpoint(checkpoint_path, generator, discriminator, device)
    
    # Test 1: Generate samples
    print("\n1. Generating random samples...")
    num_samples = 64
    fake_images = generate_samples(generator, num_samples, noise_dim, device)
    
    # Plot generated samples
    plt.figure(figsize=(10, 10))
    grid = vutils.make_grid(fake_images, nrow=8, padding=2, normalize=True)
    plt.imshow(np.transpose(grid.cpu(), (1, 2, 0)), cmap='gray')
    plt.title(f"Generated Samples (Epoch {epoch})")
    plt.axis('off')
    plt.savefig(f"{test_output_dir}/generated_samples.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test 2: Latent space interpolation
    print("\n2. Performing latent space interpolation...")
    interpolated_images = interpolate_in_latent_space(generator, noise_dim, device, steps=10)
    
    plt.figure(figsize=(15, 3))
    for i in range(interpolated_images.size(0)):
        plt.subplot(1, 10, i + 1)
        plt.imshow(interpolated_images[i, 0].cpu(), cmap='gray')
        plt.axis('off')
        plt.title(f'Step {i+1}')
    
    plt.suptitle("Latent Space Interpolation")
    plt.tight_layout()
    plt.savefig(f"{test_output_dir}/interpolation.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test 3: Compare with real images
    print("\n3. Comparing real vs generated images...")
    dataloader = prepare_mnist_dataset(batch_size=64)
    real_batch = get_sample_batch(dataloader)
    real_images = real_batch[0][:16]
    
    # Generate 16 fake images for comparison
    fake_images_comparison = generate_samples(generator, 16, noise_dim, device)
    
    plt.figure(figsize=(15, 8))
    
    # Plot real images
    plt.subplot(1, 2, 1)
    real_grid = vutils.make_grid(real_images, nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(real_grid, (1, 2, 0)), cmap='gray')
    plt.title("Real MNIST Images")
    plt.axis('off')
    
    # Plot fake images
    plt.subplot(1, 2, 2)
    fake_grid = vutils.make_grid(fake_images_comparison, nrow=4, padding=2, normalize=True)
    plt.imshow(np.transpose(fake_grid.cpu(), (1, 2, 0)), cmap='gray')
    plt.title("Generated Images")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{test_output_dir}/comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Test 4: Evaluate discriminator
    print("\n4. Evaluating discriminator performance...")
    real_accuracy = evaluate_discriminator(discriminator, dataloader, device, num_batches=10)
    
    print("\nTesting completed!")
    print(f"All test outputs saved to: {test_output_dir}")

if __name__ == '__main__':
    test_vanilla_gan()
