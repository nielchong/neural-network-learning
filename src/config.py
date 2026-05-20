import os

# ── Product ───────────────────────────────────────────────────────────────────
PRODUCT = "fish_sauce"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "price_history.csv")
MODEL_DIR = "models"

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.8
WINDOW_SIZE = 7

# ── Model Architecture ────────────────────────────────────────────────────────
INPUT_SIZE = 1
HIDDEN_SIZE = 32
NUM_LAYERS = 2
DROPOUT = 0.2

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SCHEDULER_PATIENCE = 10
SCHEDULER_FACTOR = 0.5

# ── Evaluation ────────────────────────────────────────────────────────────────
THRESHOLD_PERCENTILE = 95