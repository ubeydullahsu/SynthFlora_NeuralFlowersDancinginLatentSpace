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
    

class Decoder:
    def __init__(self, latent_dim, hidden_dim, output_dim):
        """
        Initialize the Decoder part of the VAE.
        Args:
            latent_dim (int): Dimension of the latent space.
            hidden_dim (int): Dimension of the hidden layer.
            output_dim (int): Dimension of the output data.
        """
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights and biases for the decoder here
        self.W1 = np.random.randn(hidden_dim, latent_dim) * 0.01
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.01
        self.b2 = np.zeros((output_dim, 1))

    def forward(self, z):
        """
        Forward pass through the decoder.
        Args:
            z (np.ndarray): Latent space representation of shape (batch_size, latent_dim).
        Returns:
            np.ndarray: Reconstructed output of shape (batch_size, output_dim).
        """
        h = np.dot(self.W1, z) + self.b1
        h = np.maximum(0, h)   # ReLU activation

        x_hat = 1 / (1 + np.exp(-(np.dot(self.W2, h) + self.b2)))  # Sigmoid activation for output
        
        return x_hat


def sample_latent(mu, logvar):
    """
    Sample from the latent space using the reparameterization trick.
    Args:
        mu (np.ndarray): Mean of the latent space.
        logvar (np.ndarray): Log variance of the latent space.
    Returns:
        np.ndarray: Sampled latent vector.
    """
    std = np.exp(0.5 * logvar)
    eps = np.random.randn(*mu.shape)
    return mu + eps * std

def vae_loss(x, x_hat, mu, logvar):
    """
    Compute the VAE loss function.
    Args:       
        x (np.ndarray): Original input data.
        x_hat (np.ndarray): Reconstructed output from the decoder.
        mu (np.ndarray): Mean of the latent space.
        logvar (np.ndarray): Log variance of the latent space.  
    Returns:
        float: Computed VAE loss.
    """
    # Binary Cross-Entropy Loss
    BCE  = -np.sum(x * np.log(x_hat + 1e-10) + (1 - x) * np.log(1 - x_hat + 1e-10)) / x.shape[0]
    
    # Kullback-Leibler Divergence Loss
    KLD = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar)) / x.shape[0]

    return BCE + KLD

def encoder_backward(encoder, decoder, x, mu, logvar, z, x_hat, learning_rate=0.001):
    """
    Backward pass for the encoder.
    Args:           
        encoder (Encoder): Encoder instance.
        decoder (Decoder): Decoder instance.
        x (np.ndarray): Original input data.    
        mu (np.ndarray): Mean of the latent space.
        logvar (np.ndarray): Log variance of the latent space.
        z (np.ndarray): Sampled latent vector.  
        x_hat (np.ndarray): Reconstructed output from the decoder.
        learning_rate (float): Learning rate for the optimizer.
    """

    # Compute gradients for encoder weights and biases
    dL_dx_hat = (x_hat - x) / x.shape[0]

    # Decoder gradients
    dL_dh2 = np.dot(encoder.W2.T, dL_dx_hat)
    dL_dW2 = np.dot(dL_dh2, encoder.W1.T)
    dL_db2 = np.sum(dL_dx_hat, axis=1, keepdims=True).T

    # Encoder gradients
    dL_dz = np.dot(decoder.W1.T, dL_dh2 * (decoder.W1 > 0))  # ReLU derivative
    dL_dmu = dL_dz
    dL_dLogvar = dL_dz * (0.5 * np.exp(logvar) * (z - mu)) - 0.5 * (1 - np.exp(logvar)) # Derivative of log variance

    # Update encoder weights and biases
    encoder.W1 -= learning_rate * np.dot(dL_dmu, encoder.W1.T)
    encoder.b1 -= learning_rate * np.sum(dL_dmu, axis=1, keepdims=True).T