import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Literal, Union
import copy
from tqdm.auto import tqdm
import os


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
        train_ds = self.dataset_cls(*self._train_data)
        val_ds = self.dataset_cls(*self._val_data)
        test_ds = self.dataset_cls(*self._test_data)
        self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

    def train(self) -> dict[Literal['train', 'validation'], dict[Literal['accuracy', 'loss'], list[float]]]:
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
                print(f'Restoring weights back to epoch {best_epoch}.')
                break

            print()

        return history

    @torch.no_grad()
    def evaluate(self, loader_type: Literal['val', 'test']) -> dict[Literal['accuracy', 'loss'], list[float]]:
        val_acc, val_loss = self._valid_epoch(loader_type=loader_type)
        return {'accuracy': val_acc, 'loss': val_loss}

    @torch.no_grad()
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        logits = self.model(x)
        return logits.argmax(-1)

    def _train_epoch(self) -> tuple[float, float]:
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
            total_correct += torch.eq(logits.argmax(-1), labels_batch).sum()
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