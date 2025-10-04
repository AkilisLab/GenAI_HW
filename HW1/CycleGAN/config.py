"""
Configuration file for CycleGAN training
"""
import torch

class CycleGANConfig:
    """
    Configuration class for CycleGAN hyperparameters and settings
    """
    
    # Dataset configuration
    USE_MNIST = True  # Set to True to use MNIST dataset with custom Mnist_Dataloader
    
    # Model architecture
    IMG_CHANNELS = 3
    NUM_RESIDUALS = 6  # Use 6 for 28x28 MNIST images, 9 for 256x256 images
    IMAGE_SIZE = 28 if USE_MNIST else 256  # MNIST images are 28x28
    
    # Training hyperparameters
    NUM_EPOCHS = 25  # Reduced from 50 to prevent overfitting
    BATCH_SIZE = 4 if USE_MNIST else 1  # Larger batch size for MNIST
    LEARNING_RATE = 1e-4  # Reduced from 2e-4 to slow down discriminator learning
    NUM_WORKERS = 2
    LAMBDA_CYCLE = 5  # Reduced from 10 to balance adversarial vs cycle loss
    LAMBDA_IDENTITY = 0.1  # Reduced from 0.5 to prevent over-constraint
    BETA1 = 0.5  # Beta1 parameter for Adam optimizer
    BETA2 = 0.999  # Beta2 parameter for Adam optimizer
    
    # Training settings
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    PRINT_FREQ = 10  # Print training stats every N batches
    SAVE_FREQ = 5  # Save samples and checkpoints every N epochs
    
    # Dataset paths for image folder datasets
    TRAIN_DIR_A = "data/trainA"  # Domain A training images
    TRAIN_DIR_B = "data/trainB"  # Domain B training images
    VAL_DIR_A = "data/testA"     # Domain A validation images
    VAL_DIR_B = "data/testB"     # Domain B validation images
    
    # MNIST Dataset configuration
    MNIST_DATA_PATH = "../mnist_dataset"  # Path to MNIST dataset files
    MNIST_DOMAIN_A_DIGITS = [0, 2, 4, 6, 8]  # Even digits for domain A
    MNIST_DOMAIN_B_DIGITS = [1, 3, 5, 7, 9]  # Odd digits for domain B
    
    # Checkpoint and output directories
    CHECKPOINT_DIR = "checkpoints"
    SAMPLE_DIR = "samples"
    CHECKPOINT_GEN_H = "checkpoints/gen_h.pth.tar"
    CHECKPOINT_GEN_Z = "checkpoints/gen_z.pth.tar"
    CHECKPOINT_DISC_H = "checkpoints/disc_h.pth.tar"
    CHECKPOINT_DISC_Z = "checkpoints/disc_z.pth.tar"
    
    # Model loading and saving
    LOAD_MODEL = False
    SAVE_MODEL = True
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=" * 60)
        print("CYCLEGAN CONFIGURATION")
        print("=" * 60)
        print(f"Device: {cls.DEVICE}")
        print(f"Using MNIST: {cls.USE_MNIST}")
        if cls.USE_MNIST:
            print(f"Domain A digits: {cls.MNIST_DOMAIN_A_DIGITS}")
            print(f"Domain B digits: {cls.MNIST_DOMAIN_B_DIGITS}")
        print(f"Image size: {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Number of epochs: {cls.NUM_EPOCHS}")
        print(f"Cycle loss weight: {cls.LAMBDA_CYCLE}")
        print(f"Identity loss weight: {cls.LAMBDA_IDENTITY}")
        print("=" * 60)
