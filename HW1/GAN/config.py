"""
Configuration file for Vanilla GAN training
"""
import torch

class GANConfig:
    """
    Configuration class for Vanilla GAN hyperparameters and settings
    """
    
    # Model architecture
    NOISE_DIM = 50              # Dimension of noise vector input to generator
    IMG_SHAPE = 28 * 28         # Flattened image size (28x28 for MNIST)
    
    # Training hyperparameters
    NUM_EPOCHS = 25             # Number of training epochs
    BATCH_SIZE = 128            # Batch size for training
    LEARNING_RATE = 0.0002      # Learning rate for both generator and discriminator
    BETA1 = 0.5                 # Beta1 parameter for Adam optimizer
    BETA2 = 0.999               # Beta2 parameter for Adam optimizer
    
    # Training settings
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PRINT_FREQ = 50             # Print training stats every N batches
    SAVE_FREQ = 5               # Save samples and checkpoints every N epochs
    NUM_WORKERS = 2             # Number of dataloader workers
    
    # Loss function settings
    REAL_LABEL = 1.0            # Label for real images
    FAKE_LABEL = 0.0            # Label for fake images
    
    # Directory settings
    OUTPUT_DIR = "./runs/outputs"
    CHECKPOINT_DIR = "./runs/checkpoints"
    TEST_OUTPUT_DIR = "./test_outputs"
    
    # Dataset settings
    MNIST_PATH = "../mnist_dataset"
    
    # Random seed for reproducibility
    RANDOM_SEED = 42
    
    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=" * 50)
        print("VANILLA GAN CONFIGURATION")
        print("=" * 50)
        print(f"Device: {cls.DEVICE}")
        print(f"Noise Dimension: {cls.NOISE_DIM}")
        print(f"Image Shape: {cls.IMG_SHAPE}")
        print(f"Number of Epochs: {cls.NUM_EPOCHS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Learning Rate: {cls.LEARNING_RATE}")
        print(f"Beta1: {cls.BETA1}, Beta2: {cls.BETA2}")
        print(f"Print Frequency: {cls.PRINT_FREQ}")
        print(f"Save Frequency: {cls.SAVE_FREQ}")
        print(f"Random Seed: {cls.RANDOM_SEED}")
        print("=" * 50)
