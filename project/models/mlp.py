import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Literal, Union
import copy
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )
    def forward(self, x):
        return self.layers(x)
    

class ASLDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        assert len(self.X) == len(self.y)
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

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
        num_epochs: int,
        **early_stopping_kwargs
    ):
        """
        A Trainer class for training, validating, testing, and managing a PyTorch model.
        Provides methods for training with early stopping, evaluating, saving checkpoints,
        and predicting.

        Attributes:
            model (nn.Module): The PyTorch model to train.
            optimizer (optim.Optimizer): Optimizer class for training.
            learning_rate (float): Learning rate for the optimizer.
            loss_fn (nn.CrossEntropyLoss): Loss function for optimization.
            dataset_cls (Dataset): Dataset class for loading training, validation, and test data.
            train_data (tuple): Tuple of training features and labels.
            val_data (tuple): Tuple of validation features and labels.
            test_data (tuple): Tuple of test features and labels.
            batch_size (int): Batch size for data loaders.
            num_epochs (int): Number of epochs to train.
            early_stopper (EarlyStopping): Early stopping utility for halting training.
        """
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
        self.early_stopper: EarlyStopping = EarlyStopping(**early_stopping_kwargs)
        if not os.path.exists("weights"):
            os.makedirs("weights")

    def batch_loaders(self) -> None:
        """
        Initializes DataLoader objects for training, validation, and test datasets.

        Uses the dataset class provided during initialization to create
        DataLoader instances for each of the train, validation, and test datasets.
        """
        train_ds = self.dataset_cls(*self._train_data)
        val_ds = self.dataset_cls(*self._val_data)
        test_ds = self.dataset_cls(*self._test_data)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def train(self) -> dict[Literal['train', 'validation'], dict[Literal['accuracy', 'loss'], list[float]]]:
        """
        Trains the model for the specified number of epochs.

        Tracks training and validation accuracy and loss. Saves the best model weights
        based on validation loss and supports early stopping.

        Returns:
            dict: A dictionary containing training and validation accuracy and loss history.
        """
        # Init history
        history = {
            'train': {'accuracy': [], 'loss': []},
            'validation': {'accuracy': [], 'loss': []}
        }
        # Init best epochs and loss
        best_epoch = 0
        best_val_loss = np.inf
        # Batch loaders
        self.batch_loaders()

        # Ensure the weights directory exists
        if not os.path.exists("weights"):
            os.makedirs("weights")

        # Loop through epochs
        for epoch in range(1, self.num_epochs + 1):
            print(f"Epoch {epoch}/{self.num_epochs}:")
            train_acc, train_loss = self._train_epoch()
            val_acc, val_loss = self._valid_epoch()

            # Cache metrics and loss
            history['train']['accuracy'].append(train_acc)
            history['train']['loss'].append(train_loss)
            history['validation']['accuracy'].append(val_acc)
            history['validation']['loss'].append(val_loss)

            # Save the model's weights after validation if it's the best epoch
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'history': history
                }
                weight_path = f"weights/best_model_checkpoint.pth"
                torch.save(checkpoint, weight_path)
                print(f"Saved best model weights to {weight_path}")

            # Check for early stopping
            if self.early_stopper(self.model, val_loss):
                print(f"Early stopping triggered at epoch {epoch}.")
                if self.early_stopper.restore_best_weights:
                    print(f'Restoring weights back to epoch {best_epoch}.')
                break

            print()

        return history

    @torch.no_grad()
    def evaluate(self, loader_type: Literal['val', 'test']) -> dict[Literal['accuracy', 'loss'], list[float]]:
        """
        Evaluates the model on the specified dataset (validation or test).

        Args:
            loader_type (str): Specifies the dataset loader to evaluate ('val' or 'test').

        Returns:
            dict: A dictionary containing accuracy and loss for the specified dataset.
        """
        val_acc, val_loss = self._valid_epoch(loader_type=loader_type)
        return {'accuracy': val_acc, 'loss': val_loss}

    @torch.no_grad()
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Generates predictions for the input data.

        Args:
            x (Union[np.ndarray, torch.Tensor]): Input data to predict.

        Returns:
            torch.Tensor: Predicted class indices for the input data.
        """
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        logits = self.model(x)
        return logits.argmax(-1)
    
    def load_checkpointed_weights(self, path: str) -> None:
        """
        Loads model weights from a saved checkpoint.

        Args:
            path (str): Path to the checkpoint file.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
        """
        checkpoint_dict = torch.load(path, weights_only=False)
        weights = checkpoint_dict['model_state_dict']
        self.model.load_state_dict(weights)
        print('Successfully loaded in weights from {}.'.format(path))

    def _train_epoch(self) -> tuple[float, float]:
        """
        Trains the model for one epoch.

        Computes loss and accuracy for each batch and aggregates metrics for the entire epoch.

        Returns:
            tuple: Training accuracy and loss for the epoch.
        """
        # Initialize running metrics
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for i, (X_batch, labels_batch) in pbar:
            # Zero grad
            self.optimizer.zero_grad()

            batch_idx = i + 1

            # Forward pass
            logits = self.model(X_batch)
            loss = self.loss_fn(logits, labels_batch)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Iter metrics
            running_loss += loss.item()
            total_correct += torch.eq(logits.argmax(-1), labels_batch).sum()
            total_samples += labels_batch.size(0)

            # Get batch loss and acc
            train_acc = float(total_correct / total_samples)
            train_loss = running_loss / batch_idx

            pbar.set_description(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

        return train_acc, train_loss
        
    @torch.no_grad()
    def _valid_epoch(self, loader_type: Literal['val', 'test'] = 'val') -> tuple[float, float]:
        """
        Evaluates the model on the validation or test dataset for one epoch.

        Computes loss and accuracy for each batch and aggregates metrics for the entire dataset.

        Args:
            loader_type (str, optional): Specifies the dataset loader to evaluate ('val' or 'test').
                                         Defaults to 'val'.

        Returns:
            tuple: Accuracy and loss for the dataset.
        """
        # Initialize running metrics
        running_loss = 0.0
        total_correct = 0
        total_samples = 0
        if loader_type == 'val':
            loader = self.val_loader
        elif loader_type == 'test':
            loader = self.test_loader
        else:
            raise ValueError('Loader must be "val" or "test".')
        pbar = tqdm(enumerate(loader), total=len(loader))
        for i, (X_batch, labels_batch) in pbar:
            batch_idx = i + 1

            # Forward pass
            logits = self.model(X_batch)
            loss = self.loss_fn(logits, labels_batch)

            # Iter metrics
            running_loss += loss.item()
            total_correct += torch.eq(logits.argmax(-1), labels_batch).sum().item()
            total_samples += labels_batch.size(0)

            # Get batch loss and acc
            val_acc = float(total_correct / total_samples)
            val_loss = running_loss / batch_idx

            pbar.set_description(f'{loader_type.capitalize()} Loss: {val_loss:.4f}, {loader_type.capitalize()} Acc: {val_acc:.4f}')

        return val_acc, val_loss
    

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

    def __init__(self, patience: int = 5, min_delta: int = 0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""

    def __call__(self, model: nn.Module, val_loss: float) -> bool:
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
    

def plot_history(history: dict[Literal['train', 'validation'], dict[Literal['accuracy', 'loss'], list[float]]]) -> None:
    # Extracting data
    train_acc = history['train']['accuracy']
    train_loss = history['train']['loss']
    val_acc = history['validation']['accuracy']
    val_loss = history['validation']['loss']
    epochs = range(1, len(train_acc) + 1)

    # Plotting accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    # Plotting loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()