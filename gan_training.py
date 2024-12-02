import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import os

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )

    def forward(self, x):
        return x + self.block(x)

# Define the Generator
class Generator(nn.Module):
    def __init__(self, latent_dim, channels, image_size):
        super(Generator, self).__init__()
        self.init_size = image_size // 8  # Start resolution
        self.fc = nn.Linear(latent_dim, 128 * self.init_size ** 2)

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(128),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
            ),
            nn.Sequential(
                ResidualBlock(64),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1)
            ),
        ])

    def forward(self, z):
        out = self.fc(z).view(z.shape[0], 128, self.init_size, self.init_size)
        for block in self.upsample_blocks:
            out = block(out)
        return torch.tanh(out)

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self, channels, image_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(512 * (image_size // 16) ** 2, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)

# Training and Fine-Tuning Function
def train_gan(generator, discriminator, dataloader, latent_dim, num_epochs, device):
    # Define optimizers and loss function
    adversarial_loss = nn.MSELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=0.00005, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.00001, betas=(0.5, 0.999))

    for epoch in range(num_epochs):
        for i, real_imgs in enumerate(dataloader):
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # Labels
            valid = torch.full((batch_size, 1), 0.9, device=device)
            fake = torch.zeros(batch_size, 1, device=device)

            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
