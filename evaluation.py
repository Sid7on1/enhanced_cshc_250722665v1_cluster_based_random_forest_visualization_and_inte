import logging
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch
from scipy.stats import entropy
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationMetrics:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.metrics = {}

    def evaluate(self, dataset: Dataset, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.

        Args:
        - dataset (Dataset): The dataset to evaluate on.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        accuracy = correct / len(dataset)
        self.metrics['accuracy'] = accuracy
        self.metrics['loss'] = total_loss / len(dataloader)
        return self.metrics

    def classification_report(self, dataset: Dataset, batch_size: int = 32) -> str:
        """
        Generate a classification report for the given dataset.

        Args:
        - dataset (Dataset): The dataset to generate the report for.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - str: A string containing the classification report.
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        report = classification_report(y_true, y_pred, output_dict=True)
        return report

    def confusion_matrix(self, dataset: Dataset, batch_size: int = 32) -> np.ndarray:
        """
        Generate a confusion matrix for the given dataset.

        Args:
        - dataset (Dataset): The dataset to generate the matrix for.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - np.ndarray: A numpy array containing the confusion matrix.
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        matrix = confusion_matrix(y_true, y_pred)
        return matrix

    def entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate the entropy of the given predictions.

        Args:
        - predictions (torch.Tensor): The predictions to calculate the entropy for.

        Returns:
        - torch.Tensor: A tensor containing the entropy of the predictions.
        """
        probabilities = torch.softmax(predictions, 1)
        entropy_values = entropy(probabilities.cpu().numpy(), axis=1)
        return torch.tensor(entropy_values)

    def plot_confusion_matrix(self, matrix: np.ndarray, classes: List[str]) -> None:
        """
        Plot a confusion matrix.

        Args:
        - matrix (np.ndarray): The confusion matrix to plot.
        - classes (List[str]): The classes to use for the plot.
        """
        plt.imshow(matrix, interpolation='nearest', cmap='Blues')
        plt.title('Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

class Evaluation:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.metrics = EvaluationMetrics(model, device)

    def evaluate(self, dataset: Dataset, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.

        Args:
        - dataset (Dataset): The dataset to evaluate on.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        return self.metrics.evaluate(dataset, batch_size)

    def classification_report(self, dataset: Dataset, batch_size: int = 32) -> str:
        """
        Generate a classification report for the given dataset.

        Args:
        - dataset (Dataset): The dataset to generate the report for.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - str: A string containing the classification report.
        """
        return self.metrics.classification_report(dataset, batch_size)

    def confusion_matrix(self, dataset: Dataset, batch_size: int = 32) -> np.ndarray:
        """
        Generate a confusion matrix for the given dataset.

        Args:
        - dataset (Dataset): The dataset to generate the matrix for.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - np.ndarray: A numpy array containing the confusion matrix.
        """
        return self.metrics.confusion_matrix(dataset, batch_size)

    def entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate the entropy of the given predictions.

        Args:
        - predictions (torch.Tensor): The predictions to calculate the entropy for.

        Returns:
        - torch.Tensor: A tensor containing the entropy of the predictions.
        """
        return self.metrics.entropy(predictions)

    def plot_confusion_matrix(self, matrix: np.ndarray, classes: List[str]) -> None:
        """
        Plot a confusion matrix.

        Args:
        - matrix (np.ndarray): The confusion matrix to plot.
        - classes (List[str]): The classes to use for the plot.
        """
        self.metrics.plot_confusion_matrix(matrix, classes)

class ModelEvaluator:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.evaluation = Evaluation(model, device)

    def evaluate(self, dataset: Dataset, batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on the given dataset.

        Args:
        - dataset (Dataset): The dataset to evaluate on.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        return self.evaluation.evaluate(dataset, batch_size)

    def classification_report(self, dataset: Dataset, batch_size: int = 32) -> str:
        """
        Generate a classification report for the given dataset.

        Args:
        - dataset (Dataset): The dataset to generate the report for.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - str: A string containing the classification report.
        """
        return self.evaluation.classification_report(dataset, batch_size)

    def confusion_matrix(self, dataset: Dataset, batch_size: int = 32) -> np.ndarray:
        """
        Generate a confusion matrix for the given dataset.

        Args:
        - dataset (Dataset): The dataset to generate the matrix for.
        - batch_size (int, optional): The batch size to use. Defaults to 32.

        Returns:
        - np.ndarray: A numpy array containing the confusion matrix.
        """
        return self.evaluation.confusion_matrix(dataset, batch_size)

    def entropy(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate the entropy of the given predictions.

        Args:
        - predictions (torch.Tensor): The predictions to calculate the entropy for.

        Returns:
        - torch.Tensor: A tensor containing the entropy of the predictions.
        """
        return self.evaluation.entropy(predictions)

    def plot_confusion_matrix(self, matrix: np.ndarray, classes: List[str]) -> None:
        """
        Plot a confusion matrix.

        Args:
        - matrix (np.ndarray): The confusion matrix to plot.
        - classes (List[str]): The classes to use for the plot.
        """
        self.evaluation.plot_confusion_matrix(matrix, classes)

if __name__ == "__main__":
    # Example usage
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    dataset = MyDataset()
    evaluator = ModelEvaluator(model, device)
    metrics = evaluator.evaluate(dataset, batch_size=32)
    print(metrics)
    report = evaluator.classification_report(dataset, batch_size=32)
    print(report)
    matrix = evaluator.confusion_matrix(dataset, batch_size=32)
    print(matrix)
    entropy_values = evaluator.entropy(torch.randn(10, 10))
    print(entropy_values)
    evaluator.plot_confusion_matrix(matrix, ["class1", "class2", "class3"])