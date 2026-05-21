import os

# ── Paths ──────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))

RAW_DIR         = os.path.join(BASE_DIR, "data", "raw")
CURATED_DIR     = os.path.join(BASE_DIR, "data", "curated")
FEATURES_DIR    = os.path.join(BASE_DIR, "data", "features")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "data", "predictions")
ARTIFACTS_DIR   = os.path.join(BASE_DIR, "artifacts")

SCALER_PATH     = os.path.join(ARTIFACTS_DIR, "scaler.pkl")
MODEL_PATH      = os.path.join(ARTIFACTS_DIR, "model.pt")

# ── Parámetros de la serie ──────────────────────────────
WINDOW_SIZE     = 90    # 90 días de historia como entrada
HORIZON         = 7    # predecir los próximos 30 días
TRAIN_RATIO     = 0.70
VAL_RATIO       = 0.15
REENTRENAR      = False
# test = 15% restante

# ── Entrenamiento ───────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 20
LEARNING_RATE   = 1e-3
HIDDEN_SIZE     = 64
NUM_LAYERS      = 2
GRAD_CLIP       = 1.0