"""
main.py

Main entry point for the VAE training and visualization.

Author: Derya Kara (a.k.a Ubeydullahsu)
Created: 2025-08-06
"""

from preprocessing import preprocess_dataset
from preprocessing import dataset_path
from train import train_vae

if __name__ == "__main__":

    X, y = preprocess_dataset(dataset_path)
    train_vae(X, batch_size=64)