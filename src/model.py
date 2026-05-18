import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(__file__))
import config


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    """
    Auto-selects the best available device.
    Priority: CUDA (NVIDIA GPU) → MPS (Apple GPU) → CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ── Model ─────────────────────────────────────────────────────────────────────

class PricePredictor(nn.Module):
    """
    LSTM-based next price predictor.

    Architecture:
        Input  (batch, window_size, input_size)
          → LSTM (hidden_size, num_layers, dropout)
          → last timestep output
          → Linear(hidden_size → hidden_size // 2)
          → ReLU
          → Linear(hidden_size // 2 → 1)
          → predicted next price (batch, 1)

    Anomaly detection:
        The model is trained only on normal prices.
        When it sees an anomaly, its prediction error is large.
        anomaly_score() returns that prediction error.
    """

    def __init__(
        self,
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS,
        dropout=config.DROPOUT
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: tensor of shape (batch, window_size, input_size)

        Returns:
            predicted next price of shape (batch, 1)
        """
        lstm_out, _ = self.lstm(x)         # (batch, window_size, hidden_size)
        last_step = lstm_out[:, -1, :]     # (batch, hidden_size)
        return self.predictor(last_step)   # (batch, 1)

    def anomaly_score(self, x, y):
        """
        Calculates prediction error for a single sample.
        Large error = model was surprised = likely anomaly.

        Args:
            x: input window tensor of shape (1, window_size, input_size)
            y: real next price tensor of shape (1,)

        Returns:
            scalar tensor — absolute prediction error
        """
        predicted = self.forward(x)
        return torch.abs(predicted.squeeze() - y.squeeze())