"""
train.py

Main training script for the VAE model.                             

Module for data preprocessing utilities for the project.

Author: Derya Kara (a.k.a Ubeydullahsu)
Created: 2025-08-6
"""

from vae_model import Encoder, Decoder
from vae_model import sample_latent
from vae_model import vae_loss
from vae_model import decoder_backward
from vae_model import encoder_backward
from visualize import visualize_latent_space, generate_flower, interpolate
import numpy as np

# Hyperparameters
input_dim = 2352  # 28x28x3
hidden_dim = 64
latent_dim = 2
lr = 0.001
epochs = 50

def train_vae(train_data, batch_size=64):
    """
    Train the Variational Autoencoder (VAE) model. 
    Args:
        train_data (np.ndarray): Training data of shape (num_samples, input_dim).
    Returns:
        model (VAE): Trained VAE model.
    """ 
    

    # Initialize encoder and decoder
    encoder = Encoder(input_dim, hidden_dim, latent_dim)
    decoder = Decoder(latent_dim, hidden_dim, input_dim)

    for epoch in range(epochs):
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            # Forward pass through the encoder
            mu, logvar = encoder.forward(batch)
            z = sample_latent(mu, logvar)
            x_hat = decoder.forward(z)

            # Compute loss (reconstruction + KL divergence)
            loss = vae_loss(batch, x_hat, mu, logvar)

            # Backpropagation and optimization step
            encoder_backward(encoder, decoder, batch, mu, logvar, z, x_hat, lr)
            decoder_backward(decoder, batch, z, x_hat, lr)

        # visualize latent space
        if epoch % 10 == 0:
            visualize_latent_space(encoder, train_data)
            print(f"Epoch {epoch}, Loss: {loss}")

        # Generate and visualize a flower from the latent space
        if epoch % 5 == 0:
            z_sample = np.random.randn(latent_dim)
            generate_flower(decoder, z_sample)

