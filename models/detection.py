import torch
import os
import json
from PIL import Image
from sklearn.manifold import TSNE
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_channels=3, num_params=77, latent_dim=256):
        super().__init__()
        # Image processing layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Output: [batch, 64, 128, 128]
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # Output: [batch, 128, 64, 64]
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # Output: [batch, 256, 32, 32]
            nn.ReLU()
        )

    def forward(self, x, params):
        x = self.conv_layers(x)

        return x

class ModelTrainer:
    def __init__(self, model, train_dataset, val_dataset=None, batch_size=32, learning_rate=1e-3,
                 device="cuda" if torch.cuda.is_available() else "cpu", lr_step_size=10):

        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.step_size = lr_step_size

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset else None
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.MSELoss()
        self.lr_scheduler = True if lr_step_size > 0 else False

        self.mse_loss = torch.nn.MSELoss()

        self.loss_history = []

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        for batch_idx, (idle_images, perturbed_images, params) in enumerate(self.train_loader):
            idle_images, perturbed_images, params = idle_images.to(self.device), perturbed_images.to(self.device), params.to(self.device)

            self.optimizer.zero_grad()
            recon_images, mu, log_var = self.model(idle_images, params)
            loss = self.model.loss_function(recon_images, perturbed_images, mu, log_var)
            # vgg_loss = self.perceptual_loss(recon_images, perturbed_images)
            # loss = loss + self.vgg_loss_weight * vgg_loss
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * idle_images.size(0)
        epoch_loss = running_loss / len(self.train_loader.dataset)
        self.loss_history.append(epoch_loss)

        return epoch_loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            epoch_loss = self.train_epoch(epoch)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')
            # Step the scheduler after each epoch
            if self.lr_scheduler:
                self.scheduler.step()
        self.plot_losses()

    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss - Variational Autoencoder')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plt.grid(True)
        # plt.show()
        # plt.ylim(1.7, 4)
        plt.savefig(f'plot1.png')  # Save plot as PNG file
        plt.close()  # Close the plot to free up memory

        plt.figure(figsize=(10, 5))
        plt.plot(self.loss_history, label='Training Loss')
        plt.title('Training Loss - Variational Autoencoder')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        # plt.grid(True)
        # plt.show()
        plt.ylim(7.21e6, 7.24e6)
        plt.savefig(f'plot2.png')  # Save plot as PNG file
        plt.close()  # Close the plot to free up memory


class Inference:

    @staticmethod
    def inference_display_results(output_image):
        plt.imshow(output_image)
        plt.title('Output Detection after inference')
        plt.axis('off')
        plt.show()

