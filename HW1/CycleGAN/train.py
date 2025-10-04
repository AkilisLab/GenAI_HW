"""
CycleGAN Training Script
Clean implementation following the GAN directory style
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import time
import os

# Import our custom modules
from CycleGan_Generator import Generator
from CycleGan_Discriminator import Discriminator
from dataset import prepare_cyclegan_dataloader
from utils import (set_seed, weights_init, create_output_directories, 
                   save_checkpoint, load_checkpoint, save_sample_images,
                   plot_training_losses, print_training_stats)
from config import CycleGANConfig


class CycleGANTrainer:
    """
    CycleGAN Training Class with clean implementation
    """
    
    def __init__(self, config):
        """Initialize trainer with configuration"""
        self.config = config
        self.device = config.DEVICE
        
        # Set random seed for reproducibility
        set_seed(config.RANDOM_SEED)
        
        # Create output directories
        self.checkpoint_dir, self.sample_dir = create_output_directories("./")
        
        # Initialize models
        self._setup_models()
        
        # Initialize optimizers and loss functions
        self._setup_optimizers()
        self._setup_loss_functions()
        
        # Setup data loaders
        self._setup_data_loaders()
        
        # Initialize loss tracking
        self._init_loss_tracking()
        
        print(f"CycleGAN Trainer initialized on device: {self.device}")
    
    def _setup_models(self):
        """Initialize generators and discriminators"""
        # Initialize generators
        self.gen_H = Generator(
            img_channels=self.config.IMG_CHANNELS,
            num_residuals=self.config.NUM_RESIDUALS
        ).to(self.device)
        
        self.gen_Z = Generator(
            img_channels=self.config.IMG_CHANNELS,
            num_residuals=self.config.NUM_RESIDUALS
        ).to(self.device)
        
        # Initialize discriminators
        self.disc_H = Discriminator(
            in_channels=self.config.IMG_CHANNELS
        ).to(self.device)
        
        self.disc_Z = Discriminator(
            in_channels=self.config.IMG_CHANNELS
        ).to(self.device)
        
        # Apply weight initialization
        self.gen_H.apply(weights_init)
        self.gen_Z.apply(weights_init)
        self.disc_H.apply(weights_init)
        self.disc_Z.apply(weights_init)
        
        print("Models initialized and weights applied")
    
    def _setup_optimizers(self):
        """Setup optimizers for generators and discriminators"""
        # Use different learning rates for generators and discriminators
        gen_lr = self.config.LEARNING_RATE
        disc_lr = getattr(self.config, 'DISC_LEARNING_RATE', self.config.LEARNING_RATE)
        
        self.opt_gen = optim.Adam(
            list(self.gen_Z.parameters()) + list(self.gen_H.parameters()),
            lr=gen_lr,
            betas=(self.config.BETA1, self.config.BETA2)
        )
        
        self.opt_disc = optim.Adam(
            list(self.disc_H.parameters()) + list(self.disc_Z.parameters()),
            lr=disc_lr,
            betas=(self.config.BETA1, self.config.BETA2)
        )
        
        print(f"Optimizers configured - Gen LR: {gen_lr}, Disc LR: {disc_lr}")
    
    def _setup_loss_functions(self):
        """Setup loss functions"""
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        print("Loss functions configured")
    
    def _setup_data_loaders(self):
        """Setup training and validation data loaders"""
        self.train_loader, self.val_loader = prepare_cyclegan_dataloader(self.config)
        print(f"Data loaders configured - Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Validation batches: {len(self.val_loader)}")
    
    def _init_loss_tracking(self):
        """Initialize loss tracking lists"""
        self.g_losses_H = []
        self.g_losses_Z = []
        self.d_losses_H = []
        self.d_losses_Z = []
        self.cycle_losses = []
        self.identity_losses = []
    
    def train_discriminator(self, real_A, real_B, fake_A, fake_B):
        """
        Train discriminators
        
        Returns:
            Dictionary containing discriminator losses
        """
        # Train Discriminator H (for domain B)
        self.opt_disc.zero_grad()
        
        # Real images
        disc_real_H = self.disc_H(real_B)
        disc_real_H_loss = self.mse_loss(disc_real_H, torch.ones_like(disc_real_H))
        
        # Fake images (detached to avoid generator gradients)
        disc_fake_H = self.disc_H(fake_B.detach())
        disc_fake_H_loss = self.mse_loss(disc_fake_H, torch.zeros_like(disc_fake_H))
        
        # Total discriminator H loss
        disc_H_loss = disc_real_H_loss + disc_fake_H_loss
        
        # Train Discriminator Z (for domain A)
        disc_real_Z = self.disc_Z(real_A)
        disc_real_Z_loss = self.mse_loss(disc_real_Z, torch.ones_like(disc_real_Z))
        
        disc_fake_Z = self.disc_Z(fake_A.detach())
        disc_fake_Z_loss = self.mse_loss(disc_fake_Z, torch.zeros_like(disc_fake_Z))
        
        # Total discriminator Z loss
        disc_Z_loss = disc_real_Z_loss + disc_fake_Z_loss
        
        # Combined discriminator loss
        disc_loss = (disc_H_loss + disc_Z_loss) / 2
        disc_loss.backward()
        self.opt_disc.step()
        
        return {
            'disc_H_loss': disc_H_loss,
            'disc_Z_loss': disc_Z_loss,
            'disc_total_loss': disc_loss
        }
    
    def train_generators(self, real_A, real_B):
        """
        Train generators
        
        Returns:
            Dictionary containing generator losses and generated images
        """
        self.opt_gen.zero_grad()
        
        # Identity loss (optional)
        identity_loss = 0
        if self.config.LAMBDA_IDENTITY > 0:
            # G_A should be identity if real_A is fed
            identity_A = self.gen_Z(real_A)
            identity_A_loss = self.l1_loss(identity_A, real_A)
            
            # G_B should be identity if real_B is fed
            identity_B = self.gen_H(real_B)
            identity_B_loss = self.l1_loss(identity_B, real_B)
            
            identity_loss = (identity_A_loss + identity_B_loss) / 2
        
        # Generate fake images
        fake_B = self.gen_H(real_A)  # G_H(A) -> B
        fake_A = self.gen_Z(real_B)  # G_Z(B) -> A
        
        # Adversarial loss
        disc_fake_H = self.disc_H(fake_B)
        disc_fake_Z = self.disc_Z(fake_A)
        
        loss_G_H = self.mse_loss(disc_fake_H, torch.ones_like(disc_fake_H))
        loss_G_Z = self.mse_loss(disc_fake_Z, torch.ones_like(disc_fake_Z))
        
        # Cycle loss
        cycle_A = self.gen_Z(fake_B)  # G_Z(G_H(A)) -> A
        cycle_B = self.gen_H(fake_A)  # G_H(G_Z(B)) -> B
        
        cycle_A_loss = self.l1_loss(cycle_A, real_A)
        cycle_B_loss = self.l1_loss(cycle_B, real_B)
        cycle_loss = (cycle_A_loss + cycle_B_loss) / 2
        
        # Total generator loss
        gen_loss = (
            loss_G_H + loss_G_Z + 
            cycle_loss * self.config.LAMBDA_CYCLE + 
            identity_loss * self.config.LAMBDA_IDENTITY
        )
        
        gen_loss.backward()
        self.opt_gen.step()
        
        return {
            'gen_H_loss': loss_G_H,
            'gen_Z_loss': loss_G_Z,
            'cycle_loss': cycle_loss,
            'identity_loss': identity_loss,
            'gen_total_loss': gen_loss,
            'fake_A': fake_A,
            'fake_B': fake_B
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.gen_H.train()
        self.gen_Z.train()
        self.disc_H.train()
        self.disc_Z.train()
        
        epoch_start_time = time.time()
        
        for batch_idx, (real_A, real_B) in enumerate(self.train_loader):
            real_A = real_A.to(self.device)
            real_B = real_B.to(self.device)
            
            # Train generators
            gen_results = self.train_generators(real_A, real_B)
            
            # Train discriminators
            disc_results = self.train_discriminator(
                real_A, real_B, 
                gen_results['fake_A'], 
                gen_results['fake_B']
            )
            
            # Track losses
            self.g_losses_H.append(gen_results['gen_H_loss'].item())
            self.g_losses_Z.append(gen_results['gen_Z_loss'].item())
            self.d_losses_H.append(disc_results['disc_H_loss'].item())
            self.d_losses_Z.append(disc_results['disc_Z_loss'].item())
            self.cycle_losses.append(gen_results['cycle_loss'].item())
            if gen_results['identity_loss'] != 0:
                self.identity_losses.append(gen_results['identity_loss'].item())
            
            # Print training stats
            if batch_idx % self.config.PRINT_FREQ == 0:
                losses_dict = {
                    'G_H': gen_results['gen_H_loss'],
                    'G_Z': gen_results['gen_Z_loss'],
                    'D_H': disc_results['disc_H_loss'],
                    'D_Z': disc_results['disc_Z_loss'],
                    'Cycle': gen_results['cycle_loss'],
                }
                if gen_results['identity_loss'] != 0:
                    losses_dict['Identity'] = gen_results['identity_loss']
                
                print_training_stats(
                    epoch, batch_idx, len(self.train_loader), 
                    losses_dict
                )
        
        epoch_time = time.time() - epoch_start_time
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    def save_models(self, epoch):
        """Save model checkpoints"""
        if self.config.SAVE_MODEL:
            save_checkpoint(
                self.gen_H, self.opt_gen, epoch,
                os.path.join(self.checkpoint_dir, f"gen_h_epoch_{epoch:03d}.pth")
            )
            save_checkpoint(
                self.gen_Z, self.opt_gen, epoch,
                os.path.join(self.checkpoint_dir, f"gen_z_epoch_{epoch:03d}.pth")
            )
            save_checkpoint(
                self.disc_H, self.opt_disc, epoch,
                os.path.join(self.checkpoint_dir, f"disc_h_epoch_{epoch:03d}.pth")
            )
            save_checkpoint(
                self.disc_Z, self.opt_disc, epoch,
                os.path.join(self.checkpoint_dir, f"disc_z_epoch_{epoch:03d}.pth")
            )
    
    def load_models(self):
        """Load model checkpoints if available"""
        if self.config.LOAD_MODEL:
            try:
                load_checkpoint(self.config.CHECKPOINT_GEN_H, self.gen_H, device=self.device)
                load_checkpoint(self.config.CHECKPOINT_GEN_Z, self.gen_Z, device=self.device)
                load_checkpoint(self.config.CHECKPOINT_DISC_H, self.disc_H, device=self.device)
                load_checkpoint(self.config.CHECKPOINT_DISC_Z, self.disc_Z, device=self.device)
                print("Models loaded successfully")
            except Exception as e:
                print(f"Could not load models: {e}")
    
    def train(self):
        """Main training loop"""
        print("Starting CycleGAN training...")
        self.config.print_config()
        
        # Load models if specified
        self.load_models()
        
        # Training loop
        for epoch in range(self.config.NUM_EPOCHS):
            print(f"\n{'='*60}")
            print(f"EPOCH {epoch+1}/{self.config.NUM_EPOCHS}")
            print(f"{'='*60}")
            
            # Train for one epoch
            self.train_epoch(epoch)
            
            # Save sample images
            if (epoch + 1) % self.config.SAVE_FREQ == 0 and self.val_loader:
                save_sample_images(
                    self.gen_H, self.gen_Z, self.val_loader, 
                    epoch + 1, self.sample_dir, self.device
                )
            
            # Save model checkpoints
            if (epoch + 1) % self.config.SAVE_FREQ == 0:
                self.save_models(epoch + 1)
        
        # Save final training plots
        if len(self.g_losses_H) > 0:
            plot_training_losses(
                self.g_losses_H, self.g_losses_Z,
                self.d_losses_H, self.d_losses_Z,
                self.cycle_losses, self.identity_losses,
                os.path.join(self.sample_dir, "training_losses.png")
            )
        
        print("\nTraining completed!")


def main():
    """Main function to run CycleGAN training"""
    print("CycleGAN Training")
    print("=" * 60)
    
    # Check if MNIST dataset is available (if using MNIST)
    if CycleGANConfig.USE_MNIST:
        if not os.path.exists(CycleGANConfig.MNIST_DATA_PATH):
            print(f"Error: MNIST dataset not found at {CycleGANConfig.MNIST_DATA_PATH}")
            print("Please ensure the MNIST dataset files are available.")
            return
    else:
        # Check if image directories exist
        if not os.path.exists(CycleGANConfig.TRAIN_DIR_A) or not os.path.exists(CycleGANConfig.TRAIN_DIR_B):
            print(f"Error: Training directories not found:")
            print(f"  Domain A: {CycleGANConfig.TRAIN_DIR_A}")
            print(f"  Domain B: {CycleGANConfig.TRAIN_DIR_B}")
            print("Please ensure the training data directories exist.")
            return
    
    # Initialize trainer and start training
    trainer = CycleGANTrainer(CycleGANConfig)
    trainer.train()


if __name__ == "__main__":
    main()
