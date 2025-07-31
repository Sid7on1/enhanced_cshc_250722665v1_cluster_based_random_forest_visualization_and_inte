import os
import logging
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create a file handler which logs even debug messages
file_handler = RotatingFileHandler('training.log', maxBytes=1024*1024*10, backupCount=5)
file_handler.setLevel(logging.DEBUG)

# Create a console handler with a higher log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class TrainingConfig(Enum):
    EPOCHS = 10
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.0001
    MOMENTUM = 0.9

class ModelType(Enum):
    RANDOM_FOREST = 1
    NEURAL_NETWORK = 2

class TrainingPipeline(ABC):
    def __init__(self, model_type: ModelType, config: TrainingConfig):
        self.model_type = model_type
        self.config = config
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def evaluate_model(self):
        pass

    def save_model(self):
        if self.model is not None:
            torch.save(self.model.state_dict(), 'model.pth')
            logger.info('Model saved successfully')

    def load_model(self):
        if os.path.exists('model.pth'):
            self.model.load_state_dict(torch.load('model.pth'))
            logger.info('Model loaded successfully')

class RandomForestTrainingPipeline(TrainingPipeline):
    def __init__(self, config: TrainingConfig):
        super().__init__(ModelType.RANDOM_FOREST, config)

    def load_data(self):
        # Load data from CSV file
        self.X_train = pd.read_csv('train_features.csv')
        self.y_train = pd.read_csv('train_labels.csv')
        self.X_test = pd.read_csv('test_features.csv')
        self.y_test = pd.read_csv('test_labels.csv')

    def preprocess_data(self):
        # Scale data using StandardScaler
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self):
        # Train a RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Evaluate the model on the test set
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        logger.info(f'Model accuracy: {accuracy:.3f}')
        logger.info(f'Classification report:\n{classification_report(self.y_test, y_pred)}')
        logger.info(f'Confusion matrix:\n{confusion_matrix(self.y_test, y_pred)}')

class NeuralNetworkTrainingPipeline(TrainingPipeline):
    def __init__(self, config: TrainingConfig):
        super().__init__(ModelType.NEURAL_NETWORK, config)

    def load_data(self):
        # Load data from CSV file
        self.X_train = pd.read_csv('train_features.csv')
        self.y_train = pd.read_csv('train_labels.csv')
        self.X_test = pd.read_csv('test_features.csv')
        self.y_test = pd.read_csv('test_labels.csv')

    def preprocess_data(self):
        # Scale data using StandardScaler
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self):
        # Define a neural network model
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
        # Define a loss function and an optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=self.config.LEARNING_RATE, momentum=self.config.MOMENTUM)
        # Train the model
        for epoch in range(self.config.EPOCHS):
            optimizer.zero_grad()
            outputs = self.model(torch.tensor(self.X_train, dtype=torch.float32))
            loss = criterion(outputs, torch.tensor(self.y_train, dtype=torch.long))
            loss.backward()
            optimizer.step()
            logger.info(f'Epoch {epoch+1}, Loss: {loss.item()}')

    def evaluate_model(self):
        # Evaluate the model on the test set
        outputs = self.model(torch.tensor(self.X_test, dtype=torch.float32))
        _, predicted = torch.max(outputs, 1)
        accuracy = accuracy_score(self.y_test, predicted.numpy())
        logger.info(f'Model accuracy: {accuracy:.3f}')

def main():
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--model_type', type=str, choices=['random_forest', 'neural_network'], required=True)
    args = parser.parse_args()

    config = TrainingConfig()
    if args.model_type == 'random_forest':
        pipeline = RandomForestTrainingPipeline(config)
    elif args.model_type == 'neural_network':
        pipeline = NeuralNetworkTrainingPipeline(config)

    pipeline.load_data()
    pipeline.preprocess_data()
    pipeline.train_model()
    pipeline.evaluate_model()
    pipeline.save_model()

if __name__ == '__main__':
    main()