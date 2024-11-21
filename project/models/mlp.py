import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Literal, Union
import copy


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU,
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.layers(x)
    

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: type[optim.Optimizer],
        learning_rate: float,
        loss_fn: nn.CrossEntropyLoss,
        dataset_cls: Dataset,
        train_data: tuple[np.ndarray, np.ndarray],
        val_data: tuple[np.ndarray, np.ndarray],
        test_data: tuple[np.ndarray, np.ndarray],
        batch_size: int,
        num_epochs: int
    ):
        self.model = model
        self.optimizer = optimizer(params=self.model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn
        self.dataset_cls = dataset_cls
        self.batch_size = batch_size
        self.train_loader: DataLoader = None 
        self.val_loader: DataLoader = None
        self.test_loader: DataLoader = None
        self.num_epochs = num_epochs
        self._train_data = train_data
        self._val_data = val_data
        self._test_data = test_data

    def batch_loaders(self) -> None:
        train_ds = self.dataset_cls(**self._train_data)
        val_ds = self.dataset_cls(**self._val_data)
        test_ds = self.dataset_cls(**self._test_data)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def train(self) -> dict[Literal['train', 'validation'], dict[Literal['accuracy', 'loss'], list[float]]]:
        history = {
            'train': {'accuracy': [], 'loss': []},
            'validation': {'accuracy': [], 'loss': []}
        }
        raise NotImplementedError

    def evaluate(self) -> dict[Literal['accuracy', 'loss'], list[float]]:
        raise NotImplementedError

    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        raise NotImplementedError

    def _train_epoch(self) -> tuple[float, float]:
        raise NotImplementedError

    def _valid_epoch(self) -> tuple[float, float]:
        raise NotImplementedError
    

class EarlyStopping:
    """
    Early stopping utility for monitoring validation loss during training.

    Args:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        restore_best_weights (bool): Whether to restore the model to the best state when stopping.

    Attributes:
        patience (int): Number of epochs with no improvement after which training will be stopped.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        restore_best_weights (bool): Whether to restore the model to the best state when stopping.
        best_model: Copy of the model with the best validation loss.
        best_loss (float): Best validation loss observed so far.
        counter (int): Counter for the number of epochs with no improvement.
        status (str): Current status message indicating the early stopping progress.
    """

    def __init__(self, patience: int =5, min_delta: int = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model: nn.Module, val_loss: float):
        """
        Check whether to stop training based on the validation loss.

        Args:
            model: The deep learning model being monitored.
            val_loss (float): Current validation loss.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model.load_state_dict(model.state_dict())
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.status = f"Stopped on {self.counter}"
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model.state_dict())
                return True
        self.status = f"{self.counter}/{self.patience}"
        return False