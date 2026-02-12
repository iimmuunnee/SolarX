"""LSTM model for solar generation prediction with enhanced architecture."""
import torch
import torch.nn as nn
from accelerate import Accelerator
from typing import Tuple
import numpy as np
import logging

logger = logging.getLogger("solarx.model")


class SolarLSTM(nn.Module):
    """Enhanced LSTM with dropout and layer normalization."""

    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        output_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0
    ) -> None:
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden state size
            output_size: Number of outputs (default: 1)
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        super(SolarLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (batch_size, seq_length, input_size)

        Returns:
            Output predictions (batch_size, output_size)
        """
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)

        out, _ = self.lstm(x, (h0, c0))

        # Take last timestep
        last_output = out[:, -1, :]

        # Apply layer norm and dropout
        normalized = self.layer_norm(last_output)
        dropped = self.dropout(normalized)

        # Final output
        output = self.fc(dropped)
        return output


class LSTMPredictor:
    """Wrapper for LSTM model with prediction interface."""

    def __init__(
        self,
        model_path: str = None,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0
    ) -> None:
        """
        Initialize predictor with optional model loading.

        Args:
            model_path: Path to saved model checkpoint
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
        """
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        self.model = SolarLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        if model_path:
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint)
                logger.info(f"Model loaded: {model_path}")
            except Exception as e:
                logger.warning(f"Model load failed (랜덤 가중치 사용): {e}")

        self.model = self.accelerator.prepare(self.model)
        self.model.eval()

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.

        Args:
            input_data: Input sequences (batch_size, seq_length, input_size)

        Returns:
            Predictions as numpy array
        """
        if not torch.is_tensor(input_data):
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
        else:
            input_tensor = input_data

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            pred = self.model(input_tensor)

        return pred.cpu().numpy().flatten()
