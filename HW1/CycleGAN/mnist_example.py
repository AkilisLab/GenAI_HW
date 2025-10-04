#!/usr/bin/env python3
"""
MNIST CycleGAN Example
This script demonstrates how to use the cleaned-up CycleGAN with MNIST dataset
using the custom Mnist_Dataloader.py
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from dataset import MnistCycleGANDataset, get_transforms, prepare_cyclegan_dataloader
from config import CycleGANConfig

def test_mnist_dataset():
    """Test the MNIST dataset loading"""
    print("Testing MNIST CycleGAN Dataset...")
    
    # Create dataset with your custom Mnist_Dataloader
    transform = get_transforms(CycleGANConfig.IMAGE_SIZE)
    
    train_dataset = MnistCycleGANDataset(
        mnist_data_path=CycleGANConfig.MNIST_DATA_PATH,
        domain_a_digits=CycleGANConfig.MNIST_DOMAIN_A_DIGITS,  # Even digits
        domain_b_digits=CycleGANConfig.MNIST_DOMAIN_B_DIGITS,  # Odd digits
        transform=transform,
        split='train'
    )
    
    test_dataset = MnistCycleGANDataset(
        mnist_data_path=CycleGANConfig.MNIST_DATA_PATH,
        domain_a_digits=CycleGANConfig.MNIST_DOMAIN_A_DIGITS,
        domain_b_digits=CycleGANConfig.MNIST_DOMAIN_B_DIGITS,
        transform=transform,
        split='test'
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Visualize some samples
    visualize_samples(train_dataset, num_samples=5)

def test_dataloader():
    """Test the dataloader functionality"""
    print("\nTesting CycleGAN DataLoader...")
    
    try:
        train_loader, val_loader = prepare_cyclegan_dataloader(CycleGANConfig)
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Val loader: {len(val_loader) if val_loader else 0} batches")
        
        # Test a batch
        sample_batch = next(iter(train_loader))
        domain_a, domain_b = sample_batch
        print(f"Batch shapes - Domain A: {domain_a.shape}, Domain B: {domain_b.shape}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"Error testing dataloader: {e}")
        return None, None

def visualize_samples(dataset, num_samples=5):
    """Visualize sample pairs from the dataset"""
    try:
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        fig.suptitle('MNIST CycleGAN Dataset Samples\nTop: Domain A (Even digits), Bottom: Domain B (Odd digits)')
        
        for i in range(num_samples):
            # Get a sample from dataset
            domain_a_img, domain_b_img = dataset[i * 100]  # Sample every 100th image
            
            # Convert from tensor to numpy for visualization
            if isinstance(domain_a_img, torch.Tensor):
                domain_a_np = domain_a_img.permute(1, 2, 0).numpy()
                domain_b_np = domain_b_img.permute(1, 2, 0).numpy()
                
                # Denormalize (from [-1, 1] to [0, 1])
                domain_a_np = (domain_a_np + 1) / 2
                domain_b_np = (domain_b_np + 1) / 2
            else:
                domain_a_np = domain_a_img
                domain_b_np = domain_b_img
            
            # Display images (convert RGB back to grayscale for display)
            axes[0, i].imshow(domain_a_np[:,:,0], cmap='gray', vmin=0, vmax=1)
            axes[0, i].set_title(f'Domain A Sample {i+1}')
            axes[0, i].axis('off')
            
            axes[1, i].imshow(domain_b_np[:,:,0], cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_title(f'Domain B Sample {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.savefig('mnist_cyclegan_samples.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("Sample visualization saved as 'mnist_cyclegan_samples.png'")
        
    except Exception as e:
        print(f"Error visualizing samples: {e}")

def demonstrate_training_setup():
    """Demonstrate how to set up training with MNIST"""
    print("\nDemonstrating CycleGAN training setup with MNIST...")
    
    # Print configuration
    CycleGANConfig.print_config()
    
    print("\nTo train CycleGAN with MNIST:")
    print("1. Ensure MNIST dataset is in the correct path")
    print("2. Adjust config.py settings as needed")
    print("3. Run: python train.py")
    print("\nThe model will learn to translate between even and odd digits!")
    
    print("\nKey configuration settings:")
    print(f"  - Image size: {CycleGANConfig.IMAGE_SIZE}x{CycleGANConfig.IMAGE_SIZE}")
    print(f"  - Batch size: {CycleGANConfig.BATCH_SIZE}")
    print(f"  - Learning rate: {CycleGANConfig.LEARNING_RATE}")
    print(f"  - Cycle loss weight: {CycleGANConfig.LAMBDA_CYCLE}")
    print(f"  - Identity loss weight: {CycleGANConfig.LAMBDA_IDENTITY}")

if __name__ == "__main__":
    print("MNIST CycleGAN Example")
    print("=" * 50)
    
    try:
        # Test dataset loading
        test_mnist_dataset()
        
        # Test dataloader
        train_loader, val_loader = test_dataloader()
        
        # Demonstrate training setup
        demonstrate_training_setup()
        
        print("\n" + "=" * 50)
        print("Example completed successfully!")
        print("You can now run 'python train.py' to start training CycleGAN with MNIST")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nPlease make sure:")
        print("1. MNIST dataset files are in the correct path")
        print("2. All required dependencies are installed")
        print("3. The path in config.MNIST_DATA_PATH is correct")
