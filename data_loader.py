import logging
import os
import random
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, VALIDATION_SPLIT, RANDOM_SEED
from my_exceptions import InvalidImageDataError, InvalidDirectoryError

logger = logging.getLogger(__name__)

class CustomImageFolder(data.Dataset):
    """
    Custom dataset for loading images and their labels from a custom folder structure.
    Supports image augmentation and data transformations.
    """

    def __init__(self, root: str, transform=None, target_transform=None):
        """
        Initializes the CustomImageFolder dataset.

        Parameters:
        - root (str): Root directory path containing the images and label folders.
        - transform (callable, optional): Optional transform to be applied on the images.
        - target_transform (callable, optional): Optional transform to be applied on the labels.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # Initialize the dataset
        self._init_dataset()

    def _verify_files(self, img_dir: str, lbl_dir: str) -> bool:
        """
        Verifies if the image and label folders contain valid files.

        Parameters:
        - img_dir (str): Directory containing the images.
        - lbl_dir (str): Directory containing the labels.

        Returns:
        - bool: True if the folders contain valid files, False otherwise.
        """
        valid_images = os.path.isdir(img_dir) and os.listdir(img_dir)
        valid_labels = os.path.isdir(lbl_dir) and os.listdir(lbl_dir)

        return valid_images and valid_labels

    def _find_classes(self, lbl_dir: str) -> List[str]:
        """
        Finds the class names from the label folder.

        Parameters:
        - lbl_dir (str): Directory containing the labels.

        Returns:
        - List[str]: List of class names.
        """
        classes = [d.name for d in os.scandir(lbl_dir) if d.is_dir()]
        classes.sort()
        return classes

    def _make_class_to_idx(self, classes: List[str]) -> Dict[str, int]:
        """
        Creates a dictionary mapping class names to indices.

        Parameters:
        - classes (List[str]): List of class names.

        Returns:
        - Dict[str, int]: Dictionary mapping class names to indices.
        """
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        return class_to_idx

    def _find_img_files(self, img_dir: str, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
        """
        Finds all image files and their corresponding class indices.

        Parameters:
        - img_dir (str): Directory containing the images.
        - class_to_idx (Dict[str, int]): Dictionary mapping class names to indices.

        Returns:
        - List[Tuple[str, int]]: List of tuples containing image file paths and their class indices.
        """
        img_files = []
        for target in class_to_idx.keys():
            d = os.path.join(img_dir, target)
            if not os.path.isdir(d):
                continue
            for root, dirs, files in os.walk(d):
                for fname in files:
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    img_files.append(item)

        return img_files

    def _init_dataset(self):
        """
        Initializes the dataset by setting up the image and label folders,
        finding class names, and creating a mapping of class names to indices.
        """
        img_dir = os.path.join(self.root, "images")
        lbl_dir = os.path.join(self.root, "labels")

        # Verify if the image and label folders contain valid files
        if not self._verify_files(img_dir, lbl_dir):
            error_msg = "Invalid image or label directory. Please ensure both directories contain files."
            logger.error(error_msg)
            raise InvalidDirectoryError(error_msg)

        # Find the class names from the label folder
        classes = self._find_classes(lbl_dir)

        # Create a mapping of class names to indices
        self.class_to_idx = self._make_class_to_idx(classes)

        # Find all image files and their corresponding class indices
        self.imgs = self._find_img_files(img_dir, self.class_to_idx)

    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Retrieves the image and label at the specified index.

        Parameters:
        - index (int): Index of the image and label to retrieve.

        Returns:
        - Tuple[Any, int]: Image and its corresponding label.
        """
        # Get image path and class index at the specified index
        path, target = self.imgs[index]

        # Open and transform the image
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert("RGB")
            img = img.resize(IMG_SIZE)  # Resize image to specified size
            if self.transform is not None:
                img = self.transform(img)

        # Apply target transformation if provided
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
        - int: Total number of samples.
        """
        return len(self.imgs)


class DataLoader:
    """
    Data loader class for loading and batching image data.
    Supports data augmentation, shuffling, and validation set splitting.
    """

    def __init__(self, dataset_dir: str, batch_size: int, validation_split: float = 0.2, shuffle: bool = True,
                 num_workers: int = 0, seed: int = RANDOM_SEED):
        """
        Initializes the DataLoader.

        Parameters:
        - dataset_dir (str): Directory containing the image dataset.
        - batch_size (int): Number of samples per batch.
        - validation_split (float, optional): Proportion of the dataset to use for validation.
        - shuffle (bool, optional): Whether to shuffle the dataset after each epoch.
        - num_workers (int, optional): Number of worker processes for data loading.
        - seed (int, optional): Random seed for reproducibility.
        """
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.seed = seed

        # Lock for thread safety
        self._lock = threading.Lock()

        # Initialize the dataset and data loaders
        self._init_data_loaders()

    def _init_transforms(self) -> Tuple[Any, Any]:
        """
        Initializes the data transformations for images and labels.

        Returns:
        - Tuple[Any, Any]: Image and label transformation functions.
        """
        # Define your image and label transformations here
        # For example, you can use torchvision transforms
        # transform = torchvision.transforms.Compose([
        #     torchvision.transforms.RandomCrop(IMG_SIZE),
        #     torchvision.transforms.RandomHorizontalFlip(),
        #     torchvision.transforms.ToTensor(),
        # ])

        # Placeholder transformations
        image_transform = None
        label_transform = None

        return image_transform, label_transform

    def _init_dataset(self) -> CustomImageFolder:
        """
        Initializes the CustomImageFolder dataset.

        Returns:
        - CustomImageFolder: Initialized dataset.
        """
        # Initialize image and label transformations
        image_transform, label_transform = self._init_transforms()

        # Create the dataset
        dataset = CustomImageFolder(self.dataset_dir, transform=image_transform, target_transform=label_transform)

        return dataset

    def _split_dataset(self, dataset: CustomImageFolder) -> Tuple[CustomImageFolder, CustomImageFolder]:
        """
        Splits the dataset into training and validation sets based on the validation split ratio.

        Parameters:
        - dataset (CustomImageFolder): The full dataset.

        Returns:
        - Tuple[CustomImageFolder, CustomImageFolder]: Training and validation datasets.
        """
        # Determine the number of samples for the validation set
        val_size = int(len(dataset) * self.validation_split)

        # Split the dataset randomly
        random.Random(self.seed).shuffle(dataset.imgs)
        val_dataset = data.Subset(dataset, np.arange(val_size))
        train_dataset = data.Subset(dataset, np.arange(val_size, len(dataset)))

        return train_dataset, val_dataset

    def _init_data_loaders(self):
        """
        Initializes the data loaders for training and validation sets.
        """
        # Initialize the dataset
        dataset = self._init_dataset()

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = self._split_dataset(dataset)

        # Initialize data loaders
        self.train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle,
                                           num_workers=self.num_workers)
        self.val_loader = data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers)

    def get_train_loader(self) -> data.DataLoader:
        """
        Returns the training data loader.

        Returns:
        - data.DataLoader: Training data loader.
        """
        return self.train_loader

    def get_val_loader(self) -> data.DataLoader:
        """
        Returns the validation data loader.

        Returns:
        - data.DataLoader: Validation data loader.
        """
        return self.val_loader


def load_data(dataset_dir: str = DATA_DIR, batch_size: int = BATCH_SIZE, validation_split: float = VALIDATION_SPLIT,
              shuffle: bool = True, num_workers: int = 0, seed: int = RANDOM_SEED) -> DataLoader:
    """
    Loads the image data and returns a DataLoader object for training and validation.

    Parameters:
    - dataset_dir (str, optional): Directory containing the image dataset.
    - batch_size (int, optional): Number of samples per batch.
    - validation_split (float, optional): Proportion of the dataset to use for validation.
    - shuffle (bool, optional): Whether to shuffle the dataset after each epoch.
    - num_workers (int, optional): Number of worker processes for data loading.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - DataLoader: Data loader object containing training and validation data loaders.
    """
    # Initialize the data loader
    data_loader = DataLoader(dataset_dir, batch_size, validation_split, shuffle, num_workers, seed)

    return data_loader


if __name__ == "__main__":
    # Example usage
    data_loader = load_data()
    train_loader = data_loader.get_train_loader()
    val_loader = data_loader.get_val_loader()

    # Iterate over the data loaders and access the data
    for images, labels in train_loader:
        print(images.shape, labels.shape)

    for images, labels in val_loader:
        print(images.shape, labels.shape)