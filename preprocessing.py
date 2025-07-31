# -*- coding: utf-8 -*-

"""
Image Preprocessing Utilities
=============================

This module provides various image preprocessing utilities for the computer vision project.
"""

import logging
import os
import sys
import numpy as np
from typing import Tuple, List, Dict
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define constants
IMAGE_SIZE = (224, 224)
SCALER = StandardScaler()

class ImagePreprocessor:
    """
    Image Preprocessor class
    """

    def __init__(self, image_size: Tuple[int, int] = IMAGE_SIZE):
        """
        Initialize the ImagePreprocessor instance.

        Args:
            image_size (Tuple[int, int], optional): Desired image size. Defaults to (224, 224).
        """
        self.image_size = image_size

    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image to the desired size.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Resized image.
        """
        try:
            image = Image.fromarray(image)
            image = image.resize(self.image_size)
            image = np.array(image)
            return image
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return None

    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize the image to the range [0, 1].

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Normalized image.
        """
        try:
            image = image / 255.0
            return image
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            return None

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the image by resizing and normalizing it.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Preprocessed image.
        """
        try:
            image = self.resize_image(image)
            image = self.normalize_image(image)
            return image
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return None

class ImageDataset(Dataset):
    """
    Custom Image Dataset class
    """

    def __init__(self, images: List[np.ndarray], labels: List[int], transform: transforms.Compose):
        """
        Initialize the ImageDataset instance.

        Args:
            images (List[np.ndarray]): List of input images.
            labels (List[int]): List of corresponding labels.
            transform (transforms.Compose): Transformation to apply to the images.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int):
        image = self.images[index]
        label = self.labels[index]
        if self.transform:
            image = self.transform(image)
        return image, label

class ImagePreprocessingPipeline:
    """
    Image Preprocessing Pipeline class
    """

    def __init__(self, image_size: Tuple[int, int] = IMAGE_SIZE):
        """
        Initialize the ImagePreprocessingPipeline instance.

        Args:
            image_size (Tuple[int, int], optional): Desired image size. Defaults to (224, 224).
        """
        self.image_size = image_size
        self.preprocessor = ImagePreprocessor(image_size)

    def preprocess_images(self, images: List[np.ndarray]) -> List[np.ndarray]:
        """
        Preprocess a list of images.

        Args:
            images (List[np.ndarray]): List of input images.

        Returns:
            List[np.ndarray]: List of preprocessed images.
        """
        try:
            preprocessed_images = []
            for image in images:
                preprocessed_image = self.preprocessor.preprocess_image(image)
                preprocessed_images.append(preprocessed_image)
            return preprocessed_images
        except Exception as e:
            logger.error(f"Error preprocessing images: {e}")
            return []

def load_images(image_paths: List[str]) -> List[np.ndarray]:
    """
    Load a list of images from file paths.

    Args:
        image_paths (List[str]): List of file paths to the images.

    Returns:
        List[np.ndarray]: List of loaded images.
    """
    try:
        images = []
        for image_path in image_paths:
            image = np.array(Image.open(image_path))
            images.append(image)
        return images
    except Exception as e:
        logger.error(f"Error loading images: {e}")
        return []

def split_data(images: List[np.ndarray], labels: List[int]) -> Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]:
    """
    Split the data into training and testing sets.

    Args:
        images (List[np.ndarray]): List of input images.
        labels (List[int]): List of corresponding labels.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[int], List[int]]: Training images, testing images, training labels, testing labels.
    """
    try:
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
        return train_images, test_images, train_labels, test_labels
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        return [], [], [], []

def main():
    # Example usage
    image_paths = ["image1.jpg", "image2.jpg", "image3.jpg"]
    images = load_images(image_paths)
    labels = [0, 1, 1]
    train_images, test_images, train_labels, test_labels = split_data(images, labels)

    pipeline = ImagePreprocessingPipeline()
    preprocessed_train_images = pipeline.preprocess_images(train_images)
    preprocessed_test_images = pipeline.preprocess_images(test_images)

    dataset = ImageDataset(preprocessed_train_images, train_labels, transforms.Compose([transforms.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in data_loader:
        images, labels = batch
        # Process the batch
        pass

if __name__ == "__main__":
    main()