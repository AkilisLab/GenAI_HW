import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

from CycleGan_Generator import Generator
from config import CycleGANConfig
from dataset import MnistCycleGANDataset, get_transforms
from utils import load_checkpoint, denormalize_image


def load_model(checkpoint_path, device):
    """Load a trained generator model"""
    model = Generator(
        img_channels=CycleGANConfig.IMG_CHANNELS, 
        num_residuals=CycleGANConfig.NUM_RESIDUALS
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def preprocess_image(image_path, image_size=None):
    """Preprocess image for inference"""
    if image_size is None:
        image_size = CycleGANConfig.IMAGE_SIZE
        
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


def preprocess_mnist_image(mnist_image_array):
    """Preprocess MNIST image array for inference"""
    # Convert numpy array to PIL Image
    image = Image.fromarray(mnist_image_array.astype(np.uint8)).convert('RGB')
    
    transform = get_transforms(CycleGANConfig.IMAGE_SIZE, is_training=False)
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


def postprocess_image(tensor):
    """Convert tensor back to PIL image"""
    # Denormalize
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    transform = transforms.ToPILImage()
    image = transform(tensor.squeeze(0))
    return image


def translate_image(generator, input_image_path, output_path=None):
    """Translate an image using trained generator"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Preprocess
    input_tensor = preprocess_image(input_image_path).to(device)
    
    # Generate
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    
    # Postprocess
    output_image = postprocess_image(output_tensor.cpu())
    
    if output_path:
        output_image.save(output_path)
    
    return output_image


def test_generators():
    """Test both generators with sample inputs"""
    device = CycleGANConfig.DEVICE
    print(f"Using device: {device}")
    
    # Create sample generators with correct configuration
    gen_H = Generator(
        img_channels=CycleGANConfig.IMG_CHANNELS, 
        num_residuals=CycleGANConfig.NUM_RESIDUALS
    ).to(device)
    gen_Z = Generator(
        img_channels=CycleGANConfig.IMG_CHANNELS, 
        num_residuals=CycleGANConfig.NUM_RESIDUALS
    ).to(device)
    
    # Test with random input (use correct image size)
    batch_size = 2
    sample_input = torch.randn(
        batch_size, 
        CycleGANConfig.IMG_CHANNELS, 
        CycleGANConfig.IMAGE_SIZE, 
        CycleGANConfig.IMAGE_SIZE
    ).to(device)
    
    print("Testing generators...")
    
    with torch.no_grad():
        output_H = gen_H(sample_input)
        output_Z = gen_Z(sample_input)
    
    print(f"Generator H - Input: {sample_input.shape}, Output: {output_H.shape}")
    print(f"Generator Z - Input: {sample_input.shape}, Output: {output_Z.shape}")
    
    # Test cycle consistency
    with torch.no_grad():
        cycle_output = gen_Z(output_H)
    
    print(f"Cycle consistency test - Input: {sample_input.shape}, Output: {cycle_output.shape}")
    
    print("Generators are working correctly!")


def test_mnist_translation():
    """Test MNIST digit translation"""
    if not CycleGANConfig.USE_MNIST:
        print("MNIST testing skipped - USE_MNIST is False")
        return
        
    device = CycleGANConfig.DEVICE
    print(f"Testing MNIST translation on device: {device}")
    
    try:
        # Load MNIST dataset
        test_dataset = MnistCycleGANDataset(
            mnist_data_path=CycleGANConfig.MNIST_DATA_PATH,
            domain_a_digits=CycleGANConfig.MNIST_DOMAIN_A_DIGITS,
            domain_b_digits=CycleGANConfig.MNIST_DOMAIN_B_DIGITS,
            transform=get_transforms(CycleGANConfig.IMAGE_SIZE, is_training=False),
            split='test'
        )
        
        print(f"Loaded MNIST test dataset with {len(test_dataset)} samples")
        
        # Get sample images
        sample_a, sample_b = test_dataset[0]
        print(f"Sample shapes - Domain A: {sample_a.shape}, Domain B: {sample_b.shape}")
        
        # Test with trained models if available
        gen_h_path = os.path.join(CycleGANConfig.CHECKPOINT_DIR, "gen_h_epoch_025.pth")
        gen_z_path = os.path.join(CycleGANConfig.CHECKPOINT_DIR, "gen_z_epoch_025.pth")
        
        if os.path.exists(gen_h_path) and os.path.exists(gen_z_path):
            print("Found trained models - testing translation...")
            
            gen_H = Generator(
                img_channels=CycleGANConfig.IMG_CHANNELS,
                num_residuals=CycleGANConfig.NUM_RESIDUALS
            ).to(device)
            gen_Z = Generator(
                img_channels=CycleGANConfig.IMG_CHANNELS,
                num_residuals=CycleGANConfig.NUM_RESIDUALS
            ).to(device)
            
            # Load trained weights
            load_checkpoint(gen_h_path, gen_H, device=device)
            load_checkpoint(gen_z_path, gen_Z, device=device)
            
            # Test translation
            sample_a_batch = sample_a.unsqueeze(0).to(device)
            sample_b_batch = sample_b.unsqueeze(0).to(device)
            
            with torch.no_grad():
                fake_b = gen_H(sample_a_batch)
                fake_a = gen_Z(sample_b_batch)
                cycle_a = gen_Z(fake_b)
                cycle_b = gen_H(fake_a)
            
            print("Translation test completed successfully!")
            return sample_a, sample_b, fake_b.cpu(), fake_a.cpu(), cycle_a.cpu(), cycle_b.cpu()
        else:
            print("No trained models found. Train the model first.")
            return sample_a, sample_b, None, None, None, None
            
    except Exception as e:
        print(f"Error in MNIST testing: {e}")
        return None, None, None, None, None, None


def visualize_mnist_translation(real_a, real_b, fake_b=None, fake_a=None, 
                               cycle_a=None, cycle_b=None, output_dir="test_outputs"):
    """Visualize MNIST digit translation results"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Denormalize images
    real_a = denormalize_image(real_a)
    real_b = denormalize_image(real_b)
    
    if fake_b is not None:
        fake_b = denormalize_image(fake_b.squeeze(0))
    if fake_a is not None:
        fake_a = denormalize_image(fake_a.squeeze(0))
    if cycle_a is not None:
        cycle_a = denormalize_image(cycle_a.squeeze(0))
    if cycle_b is not None:
        cycle_b = denormalize_image(cycle_b.squeeze(0))
    
    # Create visualization
    if fake_b is not None and fake_a is not None:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle('CycleGAN MNIST Translation Results', fontsize=16)
        
        # Row 1: A -> B -> A
        axes[0, 0].imshow(real_a.permute(1, 2, 0)[:,:,0], cmap='gray')
        axes[0, 0].set_title('Real A (Even)')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(fake_b.permute(1, 2, 0)[:,:,0], cmap='gray')
        axes[0, 1].set_title('Fake B (A→B)')
        axes[0, 1].axis('off')
        
        if cycle_a is not None:
            axes[0, 2].imshow(cycle_a.permute(1, 2, 0)[:,:,0], cmap='gray')
            axes[0, 2].set_title('Cycle A (A→B→A)')
            axes[0, 2].axis('off')
        
        # Row 2: B -> A -> B
        axes[1, 0].imshow(real_b.permute(1, 2, 0)[:,:,0], cmap='gray')
        axes[1, 0].set_title('Real B (Odd)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(fake_a.permute(1, 2, 0)[:,:,0], cmap='gray')
        axes[1, 1].set_title('Fake A (B→A)')
        axes[1, 1].axis('off')
        
        if cycle_b is not None:
            axes[1, 2].imshow(cycle_b.permute(1, 2, 0)[:,:,0], cmap='gray')
            axes[1, 2].set_title('Cycle B (B→A→B)')
            axes[1, 2].axis('off')
    else:
        # Just show original samples
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle('MNIST Dataset Samples', fontsize=16)
        
        axes[0].imshow(real_a.permute(1, 2, 0)[:,:,0], cmap='gray')
        axes[0].set_title('Domain A (Even digits)')
        axes[0].axis('off')
        
        axes[1].imshow(real_b.permute(1, 2, 0)[:,:,0], cmap='gray')
        axes[1].set_title('Domain B (Odd digits)')
        axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mnist_translation_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to {output_dir}/mnist_translation_result.png")


def visualize_translation(input_path, gen_checkpoint, output_dir="test_outputs"):
    """Visualize image translation for regular images"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    device = CycleGANConfig.DEVICE
    
    # Load generator
    generator = load_model(gen_checkpoint, device)
    
    # Load and process input image
    input_image = Image.open(input_path).convert("RGB")
    output_image = translate_image(generator, input_path)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(input_image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    axes[1].imshow(output_image)
    axes[1].set_title("Translated Image")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/translation_result.png", dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main function for testing"""
    print("CycleGAN Inference and Testing")
    print("=" * 50)
    
    # Print configuration
    CycleGANConfig.print_config()
    
    print("\n" + "=" * 50)
    print("TESTING MODEL ARCHITECTURE")
    print("=" * 50)
    
    # Test model architecture
    test_generators()
    
    print("\n" + "=" * 50)
    print("TESTING MNIST TRANSLATION")
    print("=" * 50)
    
    # Test MNIST translation
    if CycleGANConfig.USE_MNIST:
        results = test_mnist_translation()
        if results[0] is not None:  # If we got valid results
            real_a, real_b, fake_b, fake_a, cycle_a, cycle_b = results
            visualize_mnist_translation(real_a, real_b, fake_b, fake_a, cycle_a, cycle_b)
    else:
        print("MNIST testing skipped - using image folder dataset")
        
        # Example for regular image translation
        print("\nFor regular image translation:")
        print("1. Ensure you have trained models in the checkpoints directory")
        print("2. Use visualize_translation() function with your image path")
        print("3. Example:")
        print("   visualize_translation('path/to/image.jpg', 'checkpoints/gen_h.pth.tar')")
    
    print("\n" + "=" * 50)
    print("TESTING COMPLETED")
    print("=" * 50)


if __name__ == "__main__":
    main()
