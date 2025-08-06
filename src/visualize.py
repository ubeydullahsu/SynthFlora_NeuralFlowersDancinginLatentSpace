"""
visualize.py

Visualize the latent space of the VAE model.

Author: Derya Kara (a.k.a Ubeydullahsu)
Created: 2025-08-06
"""

import matplotlib.pyplot as plt
import numpy as np

def visualize_latent_space(encoder, data):
    """
    Visualize the latent space of the VAE model.  
    Args:
        encoder (Encoder): Encoder part of the VAE.
        decoder (Decoder): Decoder part of the VAE. 
        data (np.ndarray): Data to visualize in the latent space.
        num_samples (int): Number of samples to visualize.
    Returns:
        None
    """

    mu, _ = encoder.forward(data.T)
    plt.scatter(mu[0, :], mu[1, :], alpha=0.5)
    plt.title("Latent Space Visualization")
    plt.show()


def generate_flower(decoder, z):
    """
    Generate flower images from the latent space using the decoder.
    Args:   
        decoder (Decoder): Decoder part of the VAE.
        z (np.ndarray): Latent space samples.
    Returns:
        np.ndarray: Generated images.
    """
    x_hat = decoder.forward(z.reshape(-1, 1))
    flower = (x_hat.reshape(28, 28, 3) * 255).astype(np.uint8)
    plt.imshow(flower)
    plt.axis('off')
    plt.show()
    

def interpolate(decoder, z1, z2, n=10):
    for alpha in np.linspace(0, 1, n):
        z = alpha * z1 + (1 - alpha) * z2
        generate_flower(decoder, z)
    
