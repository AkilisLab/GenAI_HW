import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from PIL import Image

from DcGan_Generator import DCGANGenerator
from DcGan_Discriminator import DCGANDiscriminator

class DCGANTester:
    def __init__(self, checkpoint_path, device='cuda'):
        """
        Initialize DCGAN Tester
        
        Args:
            checkpoint_path (str): Path to the trained model checkpoint
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_path = checkpoint_path
        
        # Model parameters (should match training configuration)
        self.z_dim = 100
        self.g_channels = 64
        self.d_channels = 64
        self.img_channels = 1  # MNIST is grayscale
        
        # Initialize models
        self.generator = DCGANGenerator(
            z_dim=self.z_dim, 
            g_channels=self.g_channels, 
            img_channels=self.img_channels
        ).to(self.device)
        
        self.discriminator = DCGANDiscriminator(
            d_channels=self.d_channels, 
            img_channels=self.img_channels
        ).to(self.device)
        
        # Load checkpoint
        self.load_checkpoint()
        
        # Set models to evaluation mode
        self.generator.eval()
        self.discriminator.eval()
        
        print(f"DCGAN Tester initialized on {self.device}")
        print(f"Loaded checkpoint from: {checkpoint_path}")

    def load_checkpoint(self):
        """Load trained model weights from checkpoint"""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        self.epoch = checkpoint.get('epoch', 'unknown')
        print(f"Loaded model from epoch: {self.epoch}")

    def generate_samples(self, num_samples=64, save_path=None, nrow=8):
        """
        Generate new samples using the trained generator
        
        Args:
            num_samples (int): Number of samples to generate
            save_path (str): Path to save the generated images
            nrow (int): Number of images per row in the grid
        
        Returns:
            torch.Tensor: Generated images tensor
        """
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(num_samples, self.z_dim, 1, 1, device=self.device)
            
            # Generate fake images
            fake_images = self.generator(noise)
            
            # Create image grid
            img_grid = vutils.make_grid(fake_images, nrow=nrow, normalize=True, padding=2)
            
            # Save if path provided
            if save_path:
                vutils.save_image(fake_images, save_path, nrow=nrow, normalize=True, padding=2)
                print(f"Generated images saved to: {save_path}")
            
            return fake_images, img_grid

    def interpolate_latent_space(self, start_noise=None, end_noise=None, steps=10, save_path=None):
        """
        Interpolate between two points in latent space
        
        Args:
            start_noise (torch.Tensor): Starting noise vector
            end_noise (torch.Tensor): Ending noise vector
            steps (int): Number of interpolation steps
            save_path (str): Path to save interpolation results
        
        Returns:
            torch.Tensor: Interpolated images
        """
        with torch.no_grad():
            # Generate random start and end points if not provided
            if start_noise is None:
                start_noise = torch.randn(1, self.z_dim, 1, 1, device=self.device)
            if end_noise is None:
                end_noise = torch.randn(1, self.z_dim, 1, 1, device=self.device)
            
            # Create interpolation steps
            alphas = torch.linspace(0, 1, steps, device=self.device)
            interpolated_images = []
            
            for alpha in alphas:
                # Linear interpolation
                interpolated_noise = (1 - alpha) * start_noise + alpha * end_noise
                generated_image = self.generator(interpolated_noise)
                interpolated_images.append(generated_image)
            
            # Concatenate all images
            all_images = torch.cat(interpolated_images, dim=0)
            
            # Create grid
            img_grid = vutils.make_grid(all_images, nrow=steps, normalize=True, padding=2)
            
            # Save if path provided
            if save_path:
                vutils.save_image(all_images, save_path, nrow=steps, normalize=True, padding=2)
                print(f"Interpolation saved to: {save_path}")
            
            return all_images, img_grid

    def evaluate_discriminator(self, real_images_path=None, num_samples=100):
        """
        Evaluate discriminator performance on real vs fake images
        
        Args:
            real_images_path (str): Path to real images for comparison
            num_samples (int): Number of samples to evaluate
        
        Returns:
            dict: Evaluation metrics
        """
        with torch.no_grad():
            # Generate fake samples
            noise = torch.randn(num_samples, self.z_dim, 1, 1, device=self.device)
            fake_images = self.generator(noise)
            
            # Get discriminator scores for fake images
            fake_scores = self.discriminator(fake_images)
            
            metrics = {
                'fake_scores_mean': fake_scores.mean().item(),
                'fake_scores_std': fake_scores.std().item(),
                'fake_scores_min': fake_scores.min().item(),
                'fake_scores_max': fake_scores.max().item(),
            }
            
            print("Discriminator Evaluation:")
            print(f"Fake images - Mean score: {metrics['fake_scores_mean']:.4f}")
            print(f"Fake images - Std score: {metrics['fake_scores_std']:.4f}")
            print(f"Fake images - Min score: {metrics['fake_scores_min']:.4f}")
            print(f"Fake images - Max score: {metrics['fake_scores_max']:.4f}")
            
            return metrics

    def generate_class_samples(self, num_per_class=10, save_path=None):
        """
        Generate samples and try to create a diverse set
        (Note: This DCGAN doesn't have class conditioning, so this generates diverse samples)
        
        Args:
            num_per_class (int): Number of samples to generate
            save_path (str): Path to save the samples
        
        Returns:
            torch.Tensor: Generated samples
        """
        total_samples = num_per_class * 10  # Generate for 10 digit classes
        
        with torch.no_grad():
            # Generate diverse samples by using different noise patterns
            all_samples = []
            
            for i in range(10):  # For each "pseudo-class"
                # Use different random seeds for diversity
                torch.manual_seed(i * 42)
                noise = torch.randn(num_per_class, self.z_dim, 1, 1, device=self.device)
                samples = self.generator(noise)
                all_samples.append(samples)
            
            # Concatenate all samples
            all_images = torch.cat(all_samples, dim=0)
            
            # Create grid with class organization
            img_grid = vutils.make_grid(all_images, nrow=num_per_class, normalize=True, padding=2)
            
            if save_path:
                vutils.save_image(all_images, save_path, nrow=num_per_class, normalize=True, padding=2)
                print(f"Class samples saved to: {save_path}")
            
            return all_images, img_grid

    def plot_generation_comparison(self, save_path=None):
        """
        Create a comparison plot showing real vs fake images
        """
        # Generate samples
        fake_images, _ = self.generate_samples(num_samples=16, nrow=4)
        
        # Convert to numpy for plotting
        fake_np = fake_images.cpu().numpy()
        
        # Create plot
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        fig.suptitle('Generated MNIST Digits', fontsize=16)
        
        for i in range(16):
            row = i // 4
            col = i % 4
            
            # Convert from CHW to HWC and remove channel dimension
            img = fake_np[i, 0, :, :]
            
            axes[row, col].imshow(img, cmap='gray')
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Sample {i+1}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Test DCGAN Model')
    parser.add_argument('--checkpoint', '-c', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', '-o', type=str, default='test_outputs',
                       help='Output directory for test results')
    parser.add_argument('--num_samples', '-n', type=int, default=64,
                       help='Number of samples to generate')
    parser.add_argument('--device', '-d', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--interpolate', action='store_true',
                       help='Generate latent space interpolation')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate discriminator performance')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize tester
    try:
        tester = DCGANTester(args.checkpoint, args.device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples...")
    save_path = os.path.join(args.output_dir, 'generated_samples.png')
    fake_images, img_grid = tester.generate_samples(
        num_samples=args.num_samples, 
        save_path=save_path
    )
    
    # Generate class-organized samples
    print("\nGenerating class-organized samples...")
    class_save_path = os.path.join(args.output_dir, 'class_samples.png')
    tester.generate_class_samples(save_path=class_save_path)
    
    # Latent space interpolation
    if args.interpolate:
        print("\nGenerating latent space interpolation...")
        interp_save_path = os.path.join(args.output_dir, 'interpolation.png')
        tester.interpolate_latent_space(save_path=interp_save_path)
    
    # Evaluate discriminator
    if args.evaluate:
        print("\nEvaluating discriminator...")
        metrics = tester.evaluate_discriminator()
    
    # Create comparison plot
    print("\nCreating comparison plot...")
    plot_save_path = os.path.join(args.output_dir, 'comparison_plot.png')
    tester.plot_generation_comparison(save_path=plot_save_path)
    
    print(f"\nTesting completed! Results saved to: {args.output_dir}")

if __name__ == "__main__":
    # For direct execution without command line arguments
    if len(os.sys.argv) == 1:
        # Find the latest checkpoint
        checkpoint_dir = "checkpoints"
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            if checkpoints:
                latest_checkpoint = sorted(checkpoints)[-1]
                checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
                
                print(f"Found checkpoint: {checkpoint_path}")
                
                # Create output directory
                output_dir = "test_outputs"
                os.makedirs(output_dir, exist_ok=True)
                
                # Initialize tester
                tester = DCGANTester(checkpoint_path)
                
                # Run tests
                print("\nGenerating 64 samples...")
                tester.generate_samples(
                    num_samples=64, 
                    save_path=os.path.join(output_dir, 'generated_samples.png')
                )
                
                print("\nGenerating latent space interpolation...")
                tester.interpolate_latent_space(
                    save_path=os.path.join(output_dir, 'interpolation.png')
                )
                
                print("\nEvaluating discriminator...")
                tester.evaluate_discriminator()
                
                print("\nCreating comparison plot...")
                tester.plot_generation_comparison(
                    save_path=os.path.join(output_dir, 'comparison_plot.png')
                )
                
                print(f"\nTesting completed! Results saved to: {output_dir}")
            else:
                print("No checkpoints found in 'checkpoints' directory")
        else:
            print("Checkpoints directory not found")
    else:
        main()
