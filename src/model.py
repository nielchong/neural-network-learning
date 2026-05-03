import torch
import torch.nn as nn


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class PricePredictor(nn.Module):
    """
    LSTM that predicts the next price given a window of past prices.

    During training: learns on normal prices only.
    During inference: large prediction error = likely anomaly.

    This works better than an autoencoder for single-day spikes
    because the error is focused on one specific predicted value
    rather than averaged across a whole sequence.
    """

    def __init__(self, input_size=1, hidden_size=32, num_layers=2):
        super(PricePredictor, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """
        x shape: (batch_size, window_size, 1)
        Returns predicted next price: (batch_size, 1)
        """
        lstm_out, _ = self.lstm(x)
        # Use only the last timestep output
        last_output = lstm_out[:, -1, :]
        prediction = self.predictor(last_output)
        return prediction

    def anomaly_score(self, x, actual_next):
        """
        Returns prediction error for the next price.
        Higher error = more likely anomaly.
        """
        predicted = self.forward(x)
        error = torch.abs(predicted - actual_next)
        return error.squeeze()