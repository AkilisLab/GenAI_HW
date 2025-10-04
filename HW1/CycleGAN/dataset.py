"""
Dataset handling for CycleGAN
Supports both image folder datasets and MNIST using custom Mnist_Dataloader
"""
import os
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add parent directory to path to import Mnist_Dataloader
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Mnist_Dataloader import MnistDataloader


class CycleGANDataset(Dataset):
    """
    Dataset class for CycleGAN training with image folders
    Handles unpaired image datasets from two domains
    """
    def __init__(self, root_domain_a, root_domain_b, transform=None):
        """
        Args:
            root_domain_a: Path to domain A images
            root_domain_b: Path to domain B images
            transform: Transformations to apply to images
        """
        self.root_domain_a = root_domain_a
        self.root_domain_b = root_domain_b
        self.transform = transform

        # Get list of images in each domain
        self.domain_a_images = [f for f in os.listdir(root_domain_a) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        self.domain_b_images = [f for f in os.listdir(root_domain_b)
                               if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Use the length of the longer dataset
        self.length_dataset = max(len(self.domain_a_images), len(self.domain_b_images))
        self.domain_a_len = len(self.domain_a_images)
        self.domain_b_len = len(self.domain_b_images)
        
        print(f"Domain A: {self.domain_a_len} images")
        print(f"Domain B: {self.domain_b_len} images")

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        # Use modulo to cycle through shorter dataset
        domain_a_img = self.domain_a_images[index % self.domain_a_len]
        domain_b_img = self.domain_b_images[index % self.domain_b_len]

        domain_a_path = os.path.join(self.root_domain_a, domain_a_img)
        domain_b_path = os.path.join(self.root_domain_b, domain_b_img)

        domain_a_img = Image.open(domain_a_path).convert("RGB")
        domain_b_img = Image.open(domain_b_path).convert("RGB")

        if self.transform:
            domain_a_img = self.transform(domain_a_img)
            domain_b_img = self.transform(domain_b_img)

        return domain_a_img, domain_b_img


class MnistCycleGANDataset(Dataset):
    """
    MNIST Dataset class for CycleGAN training using custom Mnist_Dataloader
    Creates two domains from MNIST digits (e.g., even vs odd digits)
    """
    def __init__(self, mnist_data_path, domain_a_digits=None, domain_b_digits=None, 
                 transform=None, split='train'):
        """
        Args:
            mnist_data_path: Path to directory containing MNIST files
            domain_a_digits: List of digits for domain A (e.g., [0,1,2,3,4])
            domain_b_digits: List of digits for domain B (e.g., [5,6,7,8,9])
            transform: Transformations to apply to images
            split: 'train' or 'test'
        """
        self.transform = transform
        self.split = split
        
        # Default domains: even vs odd digits
        if domain_a_digits is None:
            domain_a_digits = [0, 2, 4, 6, 8]  # Even digits
        if domain_b_digits is None:
            domain_b_digits = [1, 3, 5, 7, 9]  # Odd digits
            
        self.domain_a_digits = set(domain_a_digits)
        self.domain_b_digits = set(domain_b_digits)
        
        # Load MNIST data
        self._load_mnist_data(mnist_data_path)
        
        print(f"MNIST {split} - Domain A ({domain_a_digits}): {len(self.domain_a_images)} images")
        print(f"MNIST {split} - Domain B ({domain_b_digits}): {len(self.domain_b_images)} images")
        
        # Use the length of the longer dataset
        self.length_dataset = max(len(self.domain_a_images), len(self.domain_b_images))
        self.domain_a_len = len(self.domain_a_images)
        self.domain_b_len = len(self.domain_b_images)

    def _load_mnist_data(self, mnist_data_path):
        """Load and separate MNIST data by domains"""
        # Initialize MNIST dataloader
        mnist_loader = MnistDataloader(
            training_images_filepath=os.path.join(mnist_data_path, 'train-images.idx3-ubyte'),
            training_labels_filepath=os.path.join(mnist_data_path, 'train-labels.idx1-ubyte'),
            test_images_filepath=os.path.join(mnist_data_path, 't10k-images.idx3-ubyte'),
            test_labels_filepath=os.path.join(mnist_data_path, 't10k-labels.idx1-ubyte')
        )
        
        # Load data
        (x_train, y_train), (x_test, y_test) = mnist_loader.load_data()
        
        if self.split == 'train':
            images, labels = x_train, y_train
        else:
            images, labels = x_test, y_test
        
        # Separate images by domain
        self.domain_a_images = []
        self.domain_b_images = []
        
        for i, label in enumerate(labels):
            if label in self.domain_a_digits:
                self.domain_a_images.append(np.array(images[i]))
            elif label in self.domain_b_digits:
                self.domain_b_images.append(np.array(images[i]))
        
        # Convert to numpy arrays for efficient indexing
        self.domain_a_images = np.array(self.domain_a_images)
        self.domain_b_images = np.array(self.domain_b_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        # Use modulo to cycle through shorter dataset
        domain_a_idx = index % self.domain_a_len
        domain_b_idx = index % self.domain_b_len
        
        domain_a_img = self.domain_a_images[domain_a_idx]
        domain_b_img = self.domain_b_images[domain_b_idx]
        
        # Convert numpy arrays to PIL Images
        # MNIST images are grayscale, convert to RGB for CycleGAN
        domain_a_img = Image.fromarray(domain_a_img.astype(np.uint8)).convert('RGB')
        domain_b_img = Image.fromarray(domain_b_img.astype(np.uint8)).convert('RGB')

        if self.transform:
            domain_a_img = self.transform(domain_a_img)
            domain_b_img = self.transform(domain_b_img)

        return domain_a_img, domain_b_img


def get_transforms(image_size=256, is_training=True):
    """
    Get image transformations for CycleGAN
    
    Args:
        image_size: Target image size
        is_training: Whether to include augmentation (for training)
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
    ]
    
    if is_training:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transforms.Compose(transform_list)


def prepare_cyclegan_dataloader(config):
    """
    Prepare CycleGAN dataloader based on configuration
    
    Args:
        config: Configuration object or dictionary
        
    Returns:
        train_loader, val_loader (val_loader can be None)
    """
    # Get configuration values
    if hasattr(config, 'USE_MNIST'):
        use_mnist = config.USE_MNIST
        batch_size = config.BATCH_SIZE
        num_workers = config.NUM_WORKERS
        image_size = config.IMAGE_SIZE
    else:
        use_mnist = config.get('use_mnist', False)
        batch_size = config.get('batch_size', 1)
        num_workers = config.get('num_workers', 2)
        image_size = config.get('image_size', 256)
    
    # Training transforms
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    if use_mnist:
        # MNIST dataset
        if hasattr(config, 'MNIST_DATA_PATH'):
            mnist_path = config.MNIST_DATA_PATH
            domain_a = config.MNIST_DOMAIN_A_DIGITS
            domain_b = config.MNIST_DOMAIN_B_DIGITS
        else:
            mnist_path = config.get('mnist_data_path')
            domain_a = config.get('mnist_domain_a_digits', [0, 2, 4, 6, 8])
            domain_b = config.get('mnist_domain_b_digits', [1, 3, 5, 7, 9])
        
        print("Using MNIST dataset with custom Mnist_Dataloader")
        
        train_dataset = MnistCycleGANDataset(
            mnist_data_path=mnist_path,
            domain_a_digits=domain_a,
            domain_b_digits=domain_b,
            transform=train_transform,
            split='train'
        )
        
        val_dataset = MnistCycleGANDataset(
            mnist_data_path=mnist_path,
            domain_a_digits=domain_a,
            domain_b_digits=domain_b,
            transform=val_transform,
            split='test'
        )
        
    else:
        # Image folder dataset
        print("Using image folder dataset")
        
        if hasattr(config, 'TRAIN_DIR_A'):
            train_dir_a = config.TRAIN_DIR_A
            train_dir_b = config.TRAIN_DIR_B
            val_dir_a = getattr(config, 'VAL_DIR_A', None)
            val_dir_b = getattr(config, 'VAL_DIR_B', None)
        else:
            train_dir_a = config.get('train_dir_a')
            train_dir_b = config.get('train_dir_b')
            val_dir_a = config.get('val_dir_a')
            val_dir_b = config.get('val_dir_b')
        
        train_dataset = CycleGANDataset(
            root_domain_a=train_dir_a,
            root_domain_b=train_dir_b,
            transform=train_transform
        )
        
        val_dataset = None
        if val_dir_a and val_dir_b and os.path.exists(val_dir_a) and os.path.exists(val_dir_b):
            val_dataset = CycleGANDataset(
                root_domain_a=val_dir_a,
                root_domain_b=val_dir_b,
                transform=val_transform
            )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    
    return train_loader, val_loader


if __name__ == "__main__":
    """Test dataset functionality"""
    print("Testing CycleGAN Dataset Module")
    print("=" * 50)
    
    # Test with MNIST dataset
    from config import CycleGANConfig
    
    try:
        train_loader, val_loader = prepare_cyclegan_dataloader(CycleGANConfig)
        print(f"\nDataset loaded successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader) if val_loader else 0}")
        
        # Test a batch
        sample_batch = next(iter(train_loader))
        domain_a, domain_b = sample_batch
        print(f"Sample batch - Domain A: {domain_a.shape}, Domain B: {domain_b.shape}")
        
    except Exception as e:
        print(f"Error testing dataset: {e}")
        print("Please ensure MNIST dataset is available at the configured path")
