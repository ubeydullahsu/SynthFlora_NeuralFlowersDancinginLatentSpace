"""
preprocessing.py

Module for data preprocessing utilities for the project.

Author: Derya Kara (a.k.a Ubeydullahsu)
Created: 2025-07-11
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

dataset_path = r"C:\github_ws\SynthFlora_NeuralFlowersDancinginLatentSpace\data"
classes = ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def preprocess_image(image_path, target_size=(28, 28)):

    """    
    Preprocess a single image by resizing and normalizing it.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Desired size for the output image.
    Returns:
        img_array (np.ndarray): Preprocessed image as a numpy array.

    """

    # open the image file
    with Image.open(image_path) as img:

        # convert to RGB if not already
        if img.mode != 'RGB':
            img = img.convert('RGB') 

        # resize the image (Lanczos interpolation)
        img = img.resize(target_size, Image.LANCZOS)

        # convert to numpy array
        img_array = np.array(img)
        # normalize pixel values to [0, 1]
        img_array = img_array / 255.0

        return img_array
    

def apply_pastel_effect(img_array):

    """
    Apply a pastel effect to the image array.

    Args:
        img_array (np.ndarray): Input image array.
    Returns:
        np.ndarray: Image array with pastel effect applied.

    """

    # Pastel weights
    weights = np.array([1.4, 1.0, 1.2]) # increase brightness of R, keep G, increase B

    # Apply weights to each channel
    pastel_img = img_array * weights

    # Decrease saturation (add gray)
    gray = img_array.mean(axis=2, keepdims=True)
    pastel_img = pastel_img * 0.7 + gray * 0.3

    return np.clip(pastel_img, 0, 1)  # ensure values are in [0, 1]


def preprocess_dataset(dataset_path, target_size=(28, 28), save_csv=True):

    """
    Preprocess the entire dataset by resizing images and applying pastel effect.  
    Save the preprocessed data to a CSV file if specified.

    Args:
        dataset_path (str): Path to the dataset directory.
        target_size (tuple): Desired size for the output images.
        save_csv (bool): Whether to save the preprocessed data to a CSV file.
    Returns:
        X (np.ndarray)(Shape: (n_samples, 2352)): Preprocessed images as a numpy array.
        y (np.ndarray)(Shape: (n_samples,)): Labels corresponding to the images.

    """

    X, y = [], []

    # Iterate through each class directory
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)

        for img_file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            if not img_file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = os.path.join(class_path, img_file)
            img_processed = preprocess_image(img_path)

            X.append(img_processed.flatten())  # flatten to 1D array
            y.append(class_name)

    # Convert lists to numpy arrays
    X = np.array(img_processed.flatten()) # 28*28*3 = 2352
    y = np.array(classes.index(class_name) for class_name in y)

    # pastel effect
    X = np.array([apply_pastel_effect(img) for img in X])
    X = X.reshape(-1, 2352)  # ensure shape is (n_samples, 2352)

    if save_csv:
        df = pd.DataFrame(X)
        df['label'] = y
        df.to_csv(os.path.join(dataset_path, 'preprocessed_flowers.csv'), index=False)


    return X, y


def test_preprocessing():
    """
    Test the preprocessing functions with a sample dataset.
    """

    X, y = preprocess_dataset(dataset_path)
    print(f"Processed {len(X)} images with shape {X.shape} and labels {y.shape}")

    return X, y


def show_sample_images(X, y, num_samples=5):
    """
    Display a few sample images from the preprocessed dataset.
    Args:
        X (np.ndarray): Preprocessed images.
        y (np.ndarray): Corresponding labels.
        num_samples (int): Number of samples to display.
    """
    plt.figure(figsize=(10, 10))

    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        img = X[i].reshape(28, 28, 3)
        plt.imshow(img, interpolation='nearest')
        plt.title(classes[y[i]])
        plt.axis('off') 

    plt.show()

X, y = test_preprocessing()
show_sample_images(X, y)