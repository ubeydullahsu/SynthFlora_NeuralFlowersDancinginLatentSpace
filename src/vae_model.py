"""
vae_model.py

Variational Autoencoder (VAE) model implementation.

Author: Derya Kara (a.k.a Ubeydullahsu)
Created: 2025-07-11

"""

import numpy as np

class Encoder:
    def __init__(self, input_dim, hidden_dim, latent_dim):
        """
        Initialize the Encoder part of the VAE.
        Args:
            input_dim (int): Dimension of the input data.
            hidden_dim (int): Dimension of the hidden layer.
            latent_dim (int): Dimension of the latent space.
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Initialize weights and biases for the encoder here
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2_mean = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b2_mean = np.zeros((1, latent_dim))
        self.W2_logvar = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b2_logvar = np.zeros((1, latent_dim))

    def forward(self, x):
        """
        Forward pass through the encoder.
        Args:
            x (np.ndarray): Input data of shape (batch_size, input_dim).
        Returns:
            tuple: Mean and log variance of the latent space.
        """
        h = np.dot(self.W1, x.T) + self.b1
        h = np.maximum(0, h)    # ReLU activation

        mu = np.dot(self.W2_mean, h) + self.b2_mean
        logvar = np.dot(self.W2_logvar, h) + self.b2_logvar

        return mu, logvar
