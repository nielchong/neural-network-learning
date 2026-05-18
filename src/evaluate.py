import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from model import PricePredictor, get_device
from train import PriceDataset, normalise
import config


# ── Scoring ───────────────────────────────────────────────────────────────────

def compute_scores(model, dataset, device):
    """
    Runs every sample in the dataset through the model and collects
    the anomaly score and real label for each one.

    Returns:
        scores — NumPy array of anomaly scores, one per sample
        labels — NumPy array of real labels (0 = normal, 1 = anomaly)
    """
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

    return np.array(scores), np.array(labels)


# ── Threshold ─────────────────────────────────────────────────────────────────

def compute_threshold(scores, labels, percentile=config.THRESHOLD_PERCENTILE):
    """
    Calculates the anomaly threshold from normal scores only.
    Anything above this threshold gets flagged as an anomaly.

    The threshold is the Nth percentile of normal scores — meaning
    N% of normal prices fall below it. Defined by THRESHOLD_PERCENTILE
    in config.py.

    Returns:
        threshold — float, the cutoff score
    """
    normal_scores = scores[labels == 0]
    return float(np.percentile(normal_scores, percentile))


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(scores, labels, threshold):
    """
    Calculates recall, precision, and F1 score given a threshold.

    Recall    = of all real anomalies, how many did we catch?
    Precision = of everything we flagged, how many were real?
    F1        = harmonic mean of recall and precision — balances both

    Returns:
        dict of recall, precision, f1, flagged, actual, correct
    """
    flagged = scores > threshold
    actual = labels == 1
    correct = flagged & actual

    n_flagged = int(flagged.sum())
    n_actual = int(actual.sum())
    n_correct = int(correct.sum())

    recall = (n_correct / n_actual * 100) if n_actual > 0 else 0
    precision = (n_correct / n_flagged * 100) if n_flagged > 0 else 0

    # F1 is the harmonic mean of recall and precision
    # It penalises models that sacrifice one for the other
    if recall + precision > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return {
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "flagged": n_flagged,
        "actual": n_actual,
        "correct": n_correct
    }


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(scores, labels, threshold, metrics):
    """
    Prints a full evaluation report to the console.
    Covers score distribution, threshold, and performance metrics.
    """
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    print("=" * 55)
    print("EVALUATION REPORT")
    print("=" * 55)

    # Score distribution
    print("\nScore Distribution:")
    print(f"  Normal  — mean: {normal_scores.mean():.6f}  "
          f"max: {normal_scores.max():.6f}")

    if len(anomaly_scores) > 0:
        print(f"  Anomaly — mean: {anomaly_scores.mean():.6f}  "
              f"max: {anomaly_scores.max():.6f}")
    else:
        print("  No anomalies in dataset")
        return

    # Separation ratio — how much higher are anomaly scores than normal?
    # A ratio above 2 means anomaly scores are at least 2x normal scores
    separation = anomaly_scores.mean() / (normal_scores.mean() + 1e-8)
    print(f"  Separation ratio:         {separation:.2f}x")

    # Threshold
    print(f"\nThreshold ({config.THRESHOLD_PERCENTILE}th percentile "
          f"of normal): {threshold:.6f}")

    # Detection results
    print(f"\nDetection Results:")
    print(f"  Flagged as anomalies:     {metrics['flagged']}")
    print(f"  Actual anomalies:         {metrics['actual']}")
    print(f"  Correctly identified:     {metrics['correct']}")

    # Performance metrics
    print(f"\nPerformance Metrics:")
    print(f"  Recall:                   {metrics['recall']:.1f}%")
    print(f"  Precision:                {metrics['precision']:.1f}%")
    print(f"  F1 Score:                 {metrics['f1']:.1f}%")

    print("=" * 55)


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate_model(model, dataset, device):
    """
    Full evaluation pipeline.

    Computes scores → calculates threshold → computes metrics → prints report.
    Also returns the threshold so predict.py can reuse it.

    Returns:
        threshold — float, the anomaly cutoff score
        metrics   — dict of recall, precision, f1
    """
    scores, labels = compute_scores(model, dataset, device)

    if (labels == 1).sum() == 0:
        print("No anomalies in dataset — cannot evaluate.")
        return None, None

    threshold = compute_threshold(scores, labels)
    metrics = compute_metrics(scores, labels, threshold)
    print_report(scores, labels, threshold, metrics)

    return threshold, metrics


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pandas as pd
    import torch

    device = get_device()
    product = config.PRODUCT

    # Load data
    df = pd.read_csv(config.DATA_PATH)
    product_df = df[df["product"] == product].sort_values("date")

    prices = product_df["price"].values.astype(float)
    labels = product_df["is_anomaly"].values.astype(float)

    prices_norm, p_min, p_max = normalise(prices)

    import numpy as np
    prices_norm = prices_norm.astype(np.float32)
    labels = labels.astype(np.float32)

    dataset = PriceDataset(
        prices_norm, labels, config.WINDOW_SIZE, only_normal=False
    )

    model = PricePredictor().to(device)
    model_path = os.path.join(config.MODEL_DIR, f"{product}_best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    evaluate_model(model, dataset, device)
