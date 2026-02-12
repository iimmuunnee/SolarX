"""Training script with validation, early stopping, and gradient clipping."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from accelerate import Accelerator
from typing import Tuple, List
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.data_loader import SolarDataManager
from src.model import SolarLSTM
from src.logger import setup_logger
from config import Config

logger = setup_logger("solarx.train", level=logging.INFO)


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss
            model: Model to save if improved

        Returns:
            True if should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.counter} epochs without improvement")
                return True
            return False


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    accelerator: Accelerator,
    config: Config
) -> Tuple[List[float], List[float]]:
    """
    Train with validation monitoring and early stopping.

    Args:
        model: LSTM model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        criterion: Loss function
        accelerator: Accelerate wrapper
        config: Configuration object

    Returns:
        (train_losses, val_losses)
    """
    early_stopping = EarlyStopping(patience=15, min_delta=0.0001)

    train_losses = []
    val_losses = []

    logger.info(f"\n>>> Training start (Max epochs: {config.model.epochs})...")

    for epoch in range(config.model.epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            accelerator.backward(loss)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch+1}/{config.model.epochs} - "
                f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

        # Early stopping check
        if early_stopping(val_loss, accelerator.unwrap_model(model)):
            logger.info(f"Restoring best model from epoch {epoch + 1 - early_stopping.patience}")
            accelerator.unwrap_model(model).load_state_dict(early_stopping.best_model_state)
            break

    return train_losses, val_losses


def train():
    """Main training function."""
    logger.info("=" * 50)
    logger.info("SolarX Training (학습)")
    logger.info("=" * 50)

    # Load configuration
    config = Config()

    # Load data
    loader = SolarDataManager()
    train_x, train_y, test_x, test_y, _ = loader.load_and_split_standard(
        str(config.paths.data_dir),
        split_ratio=0.7  # Use 70% for train+val, 30% for test
    )

    # Create sequences
    X_train, y_train = loader.create_sequences(train_x, train_y, seq_length=config.model.seq_length)

    # Split train into train + validation (80/20)
    split_idx = int(len(X_train) * 0.8)
    X_train_split, X_val = X_train[:split_idx], X_train[split_idx:]
    y_train_split, y_val = y_train[:split_idx], y_train[split_idx:]

    # Convert to tensors
    X_train_t = torch.tensor(X_train_split, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_split, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    logger.info(f"Train tensor shape: {X_train_t.shape}")
    logger.info(f"Val tensor shape: {X_val_t.shape}")

    # Initialize accelerator and model
    accelerator = Accelerator()

    model = SolarLSTM(
        input_size=config.model.input_size,
        hidden_size=config.model.hidden_size,
        output_size=1,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config.model.learning_rate)
    criterion = nn.MSELoss()

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.model.batch_size,
        shuffle=False,  # Keep temporal order
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.model.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # Train model
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, criterion, accelerator, config
    )

    # Save model
    save_path = str(config.paths.model_path)
    unwrapped_model = accelerator.unwrap_model(model)
    torch.save(unwrapped_model.state_dict(), save_path)
    logger.info(f"Model saved: {save_path}")

    # Log final metrics
    logger.info(f"\nFinal Training Loss: {train_losses[-1]:.6f}")
    logger.info(f"Final Validation Loss: {val_losses[-1]:.6f}")


if __name__ == "__main__":
    train()
