import torch
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))
from model import PricePredictor, get_device
from train import train_model
from evaluate import evaluate_model
import config


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(product=config.PRODUCT, epochs=config.EPOCHS):
    """
    Orchestrates the full ML pipeline end to end:

        1. Train    — load data, train model, save best weights
        2. Evaluate — load best weights, compute threshold and metrics
        3. Report   — print summary of pipeline run

    This is the single entry point to go from raw CSV data
    to a trained, evaluated, production-ready model.

    Args:
        product — which product to train and evaluate
        epochs  — number of training epochs

    Returns:
        model     — trained PricePredictor with best weights loaded
        threshold — anomaly score cutoff for use in predict.py
        metrics   — dict of recall, precision, f1
    """

    print("\n" + "=" * 55)
    print(f"PIPELINE START — {product.upper()}")
    print("=" * 55 + "\n")

    device = get_device()

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    print("STEP 1 — TRAINING\n")

    model, full_dataset, p_min, p_max = train_model(
        product=product,
        epochs=epochs
    )

    if model is None:
        print("Pipeline aborted — training failed.")
        return None, None, None

    # ── Step 2: Load best weights ─────────────────────────────────────────────
    # train_model returns the model at the final epoch.
    # We reload the best saved weights before evaluating.
    print("STEP 2 — LOADING BEST WEIGHTS\n")

    model_path = os.path.join(config.MODEL_DIR, f"{product}_best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded best weights from {model_path}\n")

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    print("STEP 3 — EVALUATION\n")

    threshold, metrics = evaluate_model(model, full_dataset, device)

    if threshold is None:
        print("Pipeline aborted — evaluation failed.")
        return None, None, None

    # ── Step 4: Summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("=" * 55)
    print(f"  Product:    {product}")
    print(f"  Threshold:  {threshold:.6f}")
    print(f"  Recall:     {metrics['recall']:.1f}%")
    print(f"  Precision:  {metrics['precision']:.1f}%")
    print(f"  F1 Score:   {metrics['f1']:.1f}%")
    print(f"  Model path: {model_path}")
    print("=" * 55 + "\n")

    return model, threshold, metrics


# ── Multi-product pipeline ────────────────────────────────────────────────────

def run_all_products(epochs=config.EPOCHS):
    """
    Runs the full pipeline for every product in the dataset.
    Useful for retraining all models in one shot.

    Prints a comparison table of metrics across all products
    so you can identify which products have weak models.

    Args:
        epochs — number of training epochs per product

    Returns:
        results — dict mapping product name to (threshold, metrics)
    """
    import pandas as pd

    # Read all unique products from the CSV
    df = pd.read_csv(config.DATA_PATH)
    products = df["product"].unique().tolist()

    print(f"\nFound {len(products)} products: {products}\n")

    results = {}

    for product in products:
        model, threshold, metrics = run_pipeline(product=product, epochs=epochs)

        if metrics is not None:
            results[product] = {
                "threshold": threshold,
                "recall": metrics["recall"],
                "precision": metrics["precision"],
                "f1": metrics["f1"]
            }

    # ── Comparison table ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("ALL PRODUCTS SUMMARY")
    print("=" * 65)
    print(f"{'Product':<20} {'Recall':>8} {'Precision':>10} {'F1':>8} {'Threshold':>12}")
    print("-" * 65)

    for product, r in results.items():
        print(f"{product:<20} "
              f"{r['recall']:>7.1f}% "
              f"{r['precision']:>9.1f}% "
              f"{r['f1']:>7.1f}% "
              f"{r['threshold']:>12.6f}")

    print("=" * 65 + "\n")

    return results


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TrueInflation ML Pipeline")
    parser.add_argument(
        "--product",
        type=str,
        default=config.PRODUCT,
        help=f"Product to train (default: {config.PRODUCT}). "
             f"Use 'all' to train every product."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.EPOCHS,
        help=f"Number of training epochs (default: {config.EPOCHS})"
    )
    args = parser.parse_args()

    if args.product == "all":
        run_all_products(epochs=args.epochs)
    else:
        run_pipeline(product=args.product, epochs=args.epochs)
