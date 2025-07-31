# feature_extraction.py

import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor(nn.Module):
    """
    Feature extraction layer using PCA and mutual information.
    """
    def __init__(self, n_components: int = 10, k: int = 10):
        super(FeatureExtractor, self).__init__()
        self.pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        self.selector = SelectKBest(mutual_info_classif, k=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extraction layer.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Extracted features.
        """
        x = self.scaler.fit_transform(x)
        x = self.pca.fit_transform(x)
        x = self.selector.fit_transform(x, None)
        return torch.from_numpy(x).float()

class FeatureExtractorConfig:
    """
    Configuration class for the feature extraction layer.
    """
    def __init__(self, n_components: int = 10, k: int = 10):
        self.n_components = n_components
        self.k = k

class FeatureExtractorException(Exception):
    """
    Custom exception for feature extraction errors.
    """
    pass

class FeatureExtractorModel(nn.Module):
    """
    Feature extraction model using a combination of PCA and mutual information.
    """
    def __init__(self, config: FeatureExtractorConfig):
        super(FeatureExtractorModel, self).__init__()
        self.feature_extractor = FeatureExtractor(config.n_components, config.k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extraction model.

        Args:
        x (torch.Tensor): Input tensor.

        Returns:
        torch.Tensor: Extracted features.
        """
        return self.feature_extractor(x)

def extract_features(data: pd.DataFrame, config: FeatureExtractorConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract features from the input data using the feature extraction model.

    Args:
    data (pd.DataFrame): Input data.
    config (FeatureExtractorConfig): Configuration for the feature extraction model.

    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Extracted features and labels.
    """
    try:
        # Split data into features and labels
        features = data.drop('label', axis=1)
        labels = data['label']

        # Split data into training and testing sets
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        # Create feature extraction model
        model = FeatureExtractorModel(config)

        # Train feature extraction model
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = model(torch.from_numpy(features_train.values).float())
            loss = nn.MSELoss()(outputs, torch.from_numpy(features_train.values).float())
            loss.backward()
            optimizer.step()

        # Extract features from testing data
        features_test = model(torch.from_numpy(features_test.values).float())

        return features_test, labels_test

    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise FeatureExtractorException("Error extracting features")

def main():
    # Load data
    data = pd.read_csv('data.csv')

    # Create configuration
    config = FeatureExtractorConfig(n_components=10, k=10)

    # Extract features
    features, labels = extract_features(data, config)

    # Save extracted features
    torch.save(features, 'features.pt')
    torch.save(labels, 'labels.pt')

if __name__ == "__main__":
    main()