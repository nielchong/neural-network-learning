import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.append(os.path.dirname(__file__))
from model import PricePredictor, get_device
import config


# ── Dataset ───────────────────────────────────────────────────────────────────

class PriceDataset(Dataset):
    """
    Sliding window dataset for price sequences.

    Each sample:
        x     = window of past prices (input)
        y     = next price (target to predict)
        label = whether that next price is an anomaly
    """

    def __init__(self, prices, labels, window_size=config.WINDOW_SIZE, only_normal=False):
        self.sequences = []
        self.targets = []
        self.target_labels = []

        for i in range(len(prices) - window_size):
            window = prices[i:i + window_size]
            next_price = prices[i + window_size]
            next_label = labels[i + window_size]

            if only_normal:
                # Skip windows where the target is an anomaly
                # Model should only learn to predict normal prices
                if next_label == 1:
                    continue

            self.sequences.append(window)
            self.targets.append([next_price])
            self.target_labels.append(next_label)

        mode = "normal only" if only_normal else "all data"
        print(f"  Created {len(self.sequences)} sequences ({mode})")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(
            self.sequences[idx], dtype=torch.float32
        ).unsqueeze(-1)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        label = torch.tensor(self.target_labels[idx], dtype=torch.float32)
        return x, y, label


# ── Helpers ───────────────────────────────────────────────────────────────────

def normalise(prices):
    """
    Min-max normalisation. Scales prices to [0, 1].
    Returns normalised prices, and the original min/max for reversal.
    """
    p_min = prices.min()
    p_max = prices.max()
    return (prices - p_min) / (p_max - p_min + 1e-8), p_min, p_max


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(
    product=config.PRODUCT,
    window_size=config.WINDOW_SIZE,
    epochs=config.EPOCHS
):
    """
    Full training pipeline for one product.

    Loads data → normalises → creates datasets → trains model
    → saves best weights → returns model, full dataset, and
    normalisation parameters for downstream use.

    Returns:
        model        — trained PricePredictor instance
        full_dataset — all data including anomalies, for evaluation
        p_min        — original minimum price, for reversing normalisation
        p_max        — original maximum price, for reversing normalisation
    """

    device = get_device()
    print(f"Using device: {device}")
    print(f"Training model for: {product}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_csv(config.DATA_PATH)
    product_df = df[df["product"] == product].sort_values("date")

    prices = product_df["price"].values.astype(np.float32)
    labels = product_df["is_anomaly"].values.astype(np.float32)

    # ── Normalise ─────────────────────────────────────────────────────────────
    prices_norm, p_min, p_max = normalise(prices)
    print(f"Price range: ${p_min:.2f} – ${p_max:.2f}")

    # ── Train / validation split ──────────────────────────────────────────────
    split = int(len(prices_norm) * config.TRAIN_SPLIT)
    train_prices = prices_norm[:split]
    train_labels = labels[:split]

    print(f"Training days:   {split}")
    print(f"Validation days: {len(prices_norm) - split}\n")

    # ── Datasets ──────────────────────────────────────────────────────────────
    # Training — normal windows only so the model learns normal price behaviour
    train_dataset = PriceDataset(
        train_prices, train_labels, window_size, only_normal=True
    )

    # Full dataset — all data including anomalies, used for evaluation
    full_dataset = PriceDataset(
        prices_norm, labels, window_size, only_normal=False
    )

    if len(train_dataset) == 0:
        print("ERROR: No training sequences. Reduce window_size.")
        return None, None, None, None

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = PricePredictor().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config.SCHEDULER_PATIENCE,
        factor=config.SCHEDULER_FACTOR
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print("Training...\n")
    best_loss = float("inf")
    model_path = os.path.join(config.MODEL_DIR, f"{product}_best_model.pth")
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_x, batch_y, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            predicted = model(batch_x)
            loss = criterion(predicted, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        if (epoch + 1) % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch+1}/{epochs}]  "
                  f"Loss: {avg_loss:.6f}  LR: {lr:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)

    print(f"\nBest training loss: {best_loss:.6f}")
    print(f"Model saved to {model_path}\n")

    return model, full_dataset, p_min, p_max


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from evaluate import evaluate_model

    device = get_device()

    model, dataset, p_min, p_max = train_model()

    if model and dataset:
        model_path = os.path.join(
            config.MODEL_DIR, f"{config.PRODUCT}_best_model.pth"
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        evaluate_model(model, dataset, device)