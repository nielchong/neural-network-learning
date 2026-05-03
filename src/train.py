import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import sys
import os

sys.path.append(os.path.dirname(__file__))
from model import PricePredictor, get_device


# ── Dataset ───────────────────────────────────────────────────────────────────

class PriceDataset(Dataset):
    """
    Each sample:
      x = window of past prices (input)
      y = next price (target to predict)
      label = whether that next price is an anomaly
    """

    def __init__(self, prices, labels, window_size=7, only_normal=False):
        self.sequences = []
        self.targets = []
        self.target_labels = []

        for i in range(len(prices) - window_size):
            window = prices[i:i + window_size]
            next_price = prices[i + window_size]
            next_label = labels[i + window_size]

            if only_normal:
                # Skip if next price is an anomaly
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
    p_min = prices.min()
    p_max = prices.max()
    return (prices - p_min) / (p_max - p_min + 1e-8), p_min, p_max


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(product="fish_sauce", window_size=7, epochs=100):

    device = get_device()
    print(f"Using device: {device}")
    print(f"Training model for: {product}\n")

    # Load data
    df = pd.read_csv("data/price_history.csv")
    product_df = df[df["product"] == product].sort_values("date")

    prices = product_df["price"].values.astype(np.float32)
    labels = product_df["is_anomaly"].values.astype(np.float32)

    # Normalise
    prices_norm, p_min, p_max = normalise(prices)
    print(f"Price range: ${p_min:.2f} – ${p_max:.2f}")

    # Train/validation split
    split = int(len(prices_norm) * 0.8)
    train_prices = prices_norm[:split]
    train_labels = labels[:split]

    print(f"Training days:   {split}")
    print(f"Validation days: {len(prices_norm) - split}\n")

    # Training dataset — only windows predicting normal next prices
    train_dataset = PriceDataset(
        train_prices, train_labels, window_size, only_normal=True
    )

    # Full dataset — all data for evaluation
    full_dataset = PriceDataset(
        prices_norm, labels, window_size, only_normal=False
    )

    if len(train_dataset) == 0:
        print("ERROR: No training sequences. Reduce window_size.")
        return None, None

    train_loader = DataLoader(
        train_dataset, batch_size=16, shuffle=True
    )

    # Model
    model = PricePredictor(
        input_size=1,
        hidden_size=32,
        num_layers=2
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    # Training loop
    print("Training...\n")
    best_loss = float("inf")

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
            torch.save(
                model.state_dict(),
                f"models/{product}_best_model.pth"
            )

    print(f"\nBest loss: {best_loss:.6f}")
    print(f"Model saved to models/{product}_best_model.pth\n")

    return model, full_dataset


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, dataset, device):

    model.eval()
    scores = []
    labels = []

    with torch.no_grad():
        for i in range(len(dataset)):
            x, y, label = dataset[i]
            x = x.unsqueeze(0).to(device)
            y = y.to(device)
            score = model.anomaly_score(x, y).item()
            scores.append(score)
            labels.append(label.item())

    scores = np.array(scores)
    labels = np.array(labels)

    # Score distribution — this is the key diagnostic
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    print("Score distribution:")
    print(f"  Normal  — mean: {normal_scores.mean():.6f}  "
          f"max: {normal_scores.max():.6f}")

    if len(anomaly_scores) > 0:
        print(f"  Anomaly — mean: {anomaly_scores.mean():.6f}  "
              f"max: {anomaly_scores.max():.6f}")
    else:
        print("  No anomalies in dataset")
        return

    # Threshold from normal scores
    threshold = np.percentile(normal_scores, 95)
    print(f"\nThreshold (95th pct of normal): {threshold:.6f}")

    flagged = scores > threshold
    actual = labels == 1
    correct = flagged & actual

    print(f"Flagged as anomalies:           {flagged.sum()}")
    print(f"Actual anomalies:               {actual.sum()}")
    print(f"Correctly identified:           {correct.sum()}")

    if actual.sum() > 0:
        recall = correct.sum() / actual.sum() * 100
        precision = (correct.sum() / flagged.sum() * 100
                     if flagged.sum() > 0 else 0)
        print(f"Recall:                         {recall:.1f}%")
        print(f"Precision:                      {precision:.1f}%")


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = get_device()

    model, dataset = train_model(
        product="fish_sauce",
        window_size=7,
        epochs=100
    )

    if model and dataset:
        model.load_state_dict(torch.load(
            "models/fish_sauce_best_model.pth",
            map_location=device
        ))
        evaluate_model(model, dataset, device)