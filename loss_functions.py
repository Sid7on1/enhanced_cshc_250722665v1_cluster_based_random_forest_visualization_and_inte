# loss_functions.py

"""
Custom loss functions for the computer vision project.
"""

import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LossFunction:
    """
    Base class for custom loss functions.
    """

    def __init__(self, name: str):
        """
        Initialize the loss function.

        Args:
            name (str): Name of the loss function.
        """
        self.name = name

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the loss.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Loss values.
        """
        raise NotImplementedError

class MeanSquaredError(LossFunction):
    """
    Mean squared error loss function.
    """

    def __init__(self):
        """
        Initialize the mean squared error loss function.
        """
        super().__init__("Mean Squared Error")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the mean squared error.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Mean squared error values.
        """
        return np.mean((y_true - y_pred) ** 2)

class MeanAbsoluteError(LossFunction):
    """
    Mean absolute error loss function.
    """

    def __init__(self):
        """
        Initialize the mean absolute error loss function.
        """
        super().__init__("Mean Absolute Error")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the mean absolute error.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Mean absolute error values.
        """
        return np.mean(np.abs(y_true - y_pred))

class CrossEntropyLoss(LossFunction):
    """
    Cross entropy loss function.
    """

    def __init__(self):
        """
        Initialize the cross entropy loss function.
        """
        super().__init__("Cross Entropy")

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the cross entropy.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Cross entropy values.
        """
        return -np.mean(y_true * np.log(y_pred))

class VelocityThresholdLoss(LossFunction):
    """
    Velocity threshold loss function (from the research paper).
    """

    def __init__(self, velocity_threshold: float):
        """
        Initialize the velocity threshold loss function.

        Args:
            velocity_threshold (float): Velocity threshold value.
        """
        super().__init__("Velocity Threshold")
        self.velocity_threshold = velocity_threshold

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the velocity threshold loss.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Velocity threshold loss values.
        """
        velocity = np.abs(y_true - y_pred)
        return np.mean(np.where(velocity > self.velocity_threshold, velocity, 0))

class FlowTheoryLoss(LossFunction):
    """
    Flow theory loss function (from the research paper).
    """

    def __init__(self, flow_threshold: float):
        """
        Initialize the flow theory loss function.

        Args:
            flow_threshold (float): Flow threshold value.
        """
        super().__init__("Flow Theory")
        self.flow_threshold = flow_threshold

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Compute the flow theory loss.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Flow theory loss values.
        """
        flow = np.abs(y_true - y_pred)
        return np.mean(np.where(flow > self.flow_threshold, flow, 0))

def get_loss_function(name: str, **kwargs) -> LossFunction:
    """
    Get a loss function by name.

    Args:
        name (str): Name of the loss function.

    Returns:
        LossFunction: Loss function instance.
    """
    loss_functions = {
        "mean_squared_error": MeanSquaredError(),
        "mean_absolute_error": MeanAbsoluteError(),
        "cross_entropy": CrossEntropyLoss(),
        "velocity_threshold": VelocityThresholdLoss(**kwargs),
        "flow_theory": FlowTheoryLoss(**kwargs),
    }
    return loss_functions.get(name)

# Example usage
if __name__ == "__main__":
    # Create some sample data
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 1.9, 3.2, 4.1, 5.0])

    # Get a loss function
    loss_function = get_loss_function("velocity_threshold", velocity_threshold=0.5)

    # Compute the loss
    loss = loss_function(y_true, y_pred)
    logger.info(f"Loss: {loss}")