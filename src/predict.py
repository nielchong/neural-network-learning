import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from model import PricePredictor, get_device
import config


# ── Predictor Class ───────────────────────────────────────────────────────────

class PriceAnomalyDetector:
    """
    Wraps a trained PricePredictor for real-world inference.

    Responsibilities:
        - Load trained model weights from disk
        - Accept raw price history from a user
        - Normalise using the training min/max
        - Run the model and return an anomaly verdict

    Usage:
        detector = PriceAnomalyDetector(product="fish_sauce")
        result = detector.predict(price_history=[3.45, 3.48, 3.50, 3.47, 3.52, 3.49, 3.51],
                                  new_price=8.20)
        print(result)
    """

    def __init__(self, product=config.PRODUCT, threshold=None):
        """
        Loads the trained model and its normalisation parameters from disk.

        Args:
            product   — which product model to load
            threshold — anomaly score cutoff. If None, falls back to
                        THRESHOLD_PERCENTILE in config.py applied to
                        a fixed reference value. Ideally pass the
                        threshold returned by evaluate_model().
        """
        self.product = product
        self.device = get_device()
        self.threshold = threshold

        # Load model
        self.model = PricePredictor().to(self.device)
        model_path = os.path.join(config.MODEL_DIR, f"{product}_best_model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}. "
                f"Run train.py first."
            )

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()
        print(f"Loaded model for: {product}")

    def _normalise(self, prices, p_min, p_max):
        """Applies min-max normalisation using training min/max."""
        return (prices - p_min) / (p_max - p_min + 1e-8)

    def _denormalise(self, value, p_min, p_max):
        """Reverses normalisation back to real dollar amounts."""
        return value * (p_max - p_min) + p_min

    def predict(self, price_history, new_price):
        """
        Runs anomaly detection on a new incoming price.

        Args:
            price_history — list of recent prices (length must equal WINDOW_SIZE)
            new_price     — the new price to check for anomaly

        Returns:
            dict containing:
                is_anomaly       — True if anomaly detected
                anomaly_score    — raw model score
                threshold        — cutoff used
                predicted_price  — what the model expected (in real dollars)
                actual_price     — what actually arrived (in real dollars)
                deviation_pct    — how far off the prediction was, as a %
        """
        # Validate input length
        if len(price_history) != config.WINDOW_SIZE:
            raise ValueError(
                f"price_history must have exactly {config.WINDOW_SIZE} values. "
                f"Got {len(price_history)}."
            )

        # Normalise using the history's own min/max
        # In production this should use training min/max stored from train_model
        all_prices = np.array(price_history + [new_price], dtype=np.float32)
        p_min = all_prices.min()
        p_max = all_prices.max()

        norm_history = self._normalise(np.array(price_history, dtype=np.float32), p_min, p_max)
        norm_new = self._normalise(np.array([new_price], dtype=np.float32), p_min, p_max)

        # Build tensors
        # x shape: (1, window_size, 1) — batch of 1, 7 timesteps, 1 feature
        x = torch.tensor(norm_history, dtype=torch.float32) \
                  .unsqueeze(-1) \
                  .unsqueeze(0) \
                  .to(self.device)

        # y shape: (1,) — the real next price normalised
        y = torch.tensor(norm_new, dtype=torch.float32).to(self.device)

        # Score
        with torch.no_grad():
            score = self.model.anomaly_score(x, y).item()
            predicted_norm = self.model(x).item()

        # Convert predicted price back to real dollars
        predicted_price = self._denormalise(predicted_norm, p_min, p_max)
        deviation_pct = abs(new_price - predicted_price) / (predicted_price + 1e-8) * 100

        # Determine anomaly verdict
        # Use provided threshold if available, otherwise fall back to a warning
        if self.threshold is None:
            print("Warning: no threshold set. Call evaluate_model() first "
                  "and pass threshold to PriceAnomalyDetector().")
            is_anomaly = False
        else:
            is_anomaly = score > self.threshold

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(score, 6),
            "threshold": round(self.threshold, 6) if self.threshold else None,
            "predicted_price": round(predicted_price, 2),
            "actual_price": round(new_price, 2),
            "deviation_pct": round(deviation_pct, 1)
        }


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_prediction(result, product):
    """Prints a human-readable anomaly detection result."""
    verdict = "ANOMALY DETECTED" if result["is_anomaly"] else "Normal"

    print("=" * 45)
    print(f"Product:          {product}")
    print(f"Actual price:     ${result['actual_price']:.2f}")
    print(f"Predicted price:  ${result['predicted_price']:.2f}")
    print(f"Deviation:        {result['deviation_pct']:.1f}%")
    print(f"Anomaly score:    {result['anomaly_score']:.6f}")
    print(f"Threshold:        {result['threshold']}")
    print(f"Verdict:          {verdict}")
    print("=" * 45)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from evaluate import evaluate_model, compute_scores, compute_threshold
    from train import PriceDataset, normalise
    import pandas as pd

    device = get_device()
    product = config.PRODUCT

    # Load full dataset to compute threshold
    df = pd.read_csv(config.DATA_PATH)
    product_df = df[df["product"] == product].sort_values("date")

    prices = product_df["price"].values.astype(np.float32)
    labels = product_df["is_anomaly"].values.astype(np.float32)
    prices_norm, p_min, p_max = normalise(prices)

    dataset = PriceDataset(
        prices_norm, labels, config.WINDOW_SIZE, only_normal=False
    )

    # Load model and compute threshold from evaluation
    from model import PricePredictor
    model = PricePredictor().to(device)
    model_path = os.path.join(config.MODEL_DIR, f"{product}_best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    scores, eval_labels = compute_scores(model, dataset, device)
    threshold = compute_threshold(scores, eval_labels)
    print(f"Threshold computed from evaluation: {threshold:.6f}\n")

    # Example prediction — simulating a new receipt scan
    # Last 7 days of normal prices, then a spike
    example_history = [3.45, 3.48, 3.50, 3.47, 3.52, 3.49, 3.51]
    example_normal_price = 3.53
    example_anomaly_price = 7.20

    detector = PriceAnomalyDetector(product=product, threshold=threshold)

    print("\n--- Normal price ---")
    result = detector.predict(example_history, example_normal_price)
    print_prediction(result, product)

    print("\n--- Anomaly price ---")
    result = detector.predict(example_history, example_anomaly_price)
    print_prediction(result, product)
