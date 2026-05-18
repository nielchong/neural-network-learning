import os

# ── Product ───────────────────────────────────────────────────────────────────
# Change this to train a different product
PRODUCT = "fish_sauce"

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "price_history.csv")
MODEL_DIR = "models"

# ── Data ──────────────────────────────────────────────────────────────────────
TRAIN_SPLIT = 0.8       # 80% training, 20% validation
WINDOW_SIZE = 7         # days of history per sliding window sample

# ── Model Architecture ────────────────────────────────────────────────────────
INPUT_SIZE = 1          # features per timestep — just price for now
HIDDEN_SIZE = 32        # neurons in the LSTM layer
NUM_LAYERS = 2          # stacked LSTM layers
DROPOUT = 0.2           # fraction of neurons disabled during training

# ── Training ──────────────────────────────────────────────────────────────────
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 0.001
SCHEDULER_PATIENCE = 10     # epochs of no improvement before reducing LR
SCHEDULER_FACTOR = 0.5      # multiply LR by this when patience runs out

# ── Evaluation ────────────────────────────────────────────────────────────────
THRESHOLD_PERCENTILE = 95   # anomaly threshold = this percentile of normal scores
