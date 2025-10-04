import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add parent directory to path to import MnistDataloader
sys.path.append('..')
from Mnist_Dataloader import MnistDataloader

def prepare_mnist_dataset(batch_size=128, num_workers=2):
    """
    Prepare MNIST dataset using the custom MnistDataloader
    """
    # Define paths to MNIST dataset files
    mnist_path = '../mnist_dataset'
    training_images_filepath = os.path.join(mnist_path, 'train-images.idx3-ubyte')
    training_labels_filepath = os.path.join(mnist_path, 'train-labels.idx1-ubyte')
    test_images_filepath = os.path.join(mnist_path, 't10k-images.idx3-ubyte')
    test_labels_filepath = os.path.join(mnist_path, 't10k-labels.idx1-ubyte')
    
    # Load MNIST dataset using custom dataloader
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, 
                                      test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    # Convert to tensors and normalize
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
    x_train = x_train.unsqueeze(1)  # Add channel dimension: (N, 1, 28, 28)
    
    # Normalize to [-1, 1] range (matching Tanh output of generator)
    x_train = (x_train / 255.0 - 0.5) / 0.5
    
    # Create dataset and dataloader
    dataset = TensorDataset(x_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=True)
    
    return dataloader

def get_sample_batch(dataloader):
    """
    Get a sample batch from the dataloader for visualization
    """
    return next(iter(dataloader))
