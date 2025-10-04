import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import random

# Set random seed for reproducibility
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results

# Add parent directory to path to import MnistDataloader
sys.path.append('..')
from Mnist_Dataloader import MnistDataloader

from DcGan_Generator import DCGANGenerator
from DcGan_Discriminator import DCGANDiscriminator

# Device configuration
device = torch.device("cuda:0" if (torch.cuda.is_available() and 1 > 0) else "cpu")
print(f"Using device: {device}")

# Hyperparameters (following PyTorch DCGAN tutorial)
nz = 100          # Size of z latent vector (generator input)
ngf = 64          # Size of feature maps in generator  
ndf = 64          # Size of feature maps in discriminator
nc = 1            # Number of channels (1 for MNIST grayscale)
num_epochs = 8    # Number of training epochs
lr = 0.0002       # Learning rate for optimizers
beta1 = 0.5       # Beta1 hyperparameter for Adam optimizers
batch_size = 128  # Batch size during training
image_size = 64   # Spatial size of training images

# Create directories for outputs
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

# Load MNIST dataset using custom dataloader
mnist_path = '../mnist_dataset'
training_images_filepath = os.path.join(mnist_path, 'train-images.idx3-ubyte')
training_labels_filepath = os.path.join(mnist_path, 'train-labels.idx1-ubyte')
test_images_filepath = os.path.join(mnist_path, 't10k-images.idx3-ubyte')
test_labels_filepath = os.path.join(mnist_path, 't10k-labels.idx1-ubyte')

mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, 
                                  test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

# Convert to tensors and normalize
x_train = torch.tensor(np.array(x_train), dtype=torch.float32)
x_train = x_train.unsqueeze(1)  # Add channel dimension
x_train = torch.nn.functional.interpolate(x_train, size=(image_size, image_size), mode='bilinear', align_corners=False)
x_train = (x_train / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]

# Create dataset and dataloader
dataset = TensorDataset(x_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Initialize models (using the tutorial naming convention)
netG = DCGANGenerator(z_dim=nz, g_channels=ngf, img_channels=nc).to(device)
netD = DCGANDiscriminator(d_channels=ndf, img_channels=nc).to(device)

# Custom weights initialization called on netG and netD (from PyTorch tutorial)
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Apply the weights_init function to randomly initialize all weights
netG.apply(weights_init)
netD.apply(weights_init)

# Initialize BCELoss function
criterion = nn.BCELoss()

# Create batch of latent vectors for visualization
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop
print("Starting Training Loop...")

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

    # Save sample images every epoch
    if epoch % 1 == 0 or epoch == num_epochs - 1:
        with torch.no_grad():
            fake_imgs = netG(fixed_noise)
            vutils.save_image(fake_imgs, f"outputs/fake_samples_epoch_{epoch:03d}.png", 
                            normalize=True, nrow=8, padding=2)
        
        # Save checkpoints
        torch.save({
            'epoch': epoch,
            'generator_state_dict': netG.state_dict(),
            'discriminator_state_dict': netD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'g_loss': errG.item(),
            'd_loss': errD.item(),
        }, f"checkpoints/checkpoint_epoch_{epoch:03d}.pth")

# Plot training losses
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("outputs/training_losses.png")
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images vs fake images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.savefig("outputs/real_vs_fake_comparison.png")
plt.show()

print("Training completed!")
