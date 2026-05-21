import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.preprocessing import StandardScaler
from config import CURATED_DIR, FEATURES_DIR, ARTIFACTS_DIR
from config import WINDOW_SIZE, HORIZON, TRAIN_RATIO, VAL_RATIO

# Leer
df = pd.read_csv(os.path.join(CURATED_DIR, "volumen_m3.csv"))
valores = df["volumen_m3"].values.reshape(-1, 1)

# Split temporal
n = len(valores)
train_end = int(n * TRAIN_RATIO)
val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))

train = valores[:train_end]
val   = valores[train_end:val_end]
test  = valores[val_end:]

print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

# Normalizar solo sobre train
scaler = StandardScaler()
train_s = scaler.fit_transform(train)
val_s   = scaler.transform(val)
test_s  = scaler.transform(test)

# Guardar scaler
joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.pkl"))
print("Scaler guardado.")

# Sliding window
def sliding_window(data, L, h):
    X, y = [], []
    for i in range(len(data) - L - h + 1):
        X.append(data[i:i+L])
        y.append(data[i+L:i+L+h].flatten())
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.float32))

X_train, y_train = sliding_window(train_s, WINDOW_SIZE, HORIZON)
X_val,   y_val   = sliding_window(val_s,   WINDOW_SIZE, HORIZON)
X_test,  y_test  = sliding_window(test_s,  WINDOW_SIZE, HORIZON)

print(f"X_train: {X_train.shape} | X_val: {X_val.shape} | X_test: {X_test.shape}")

# Guardar tensores
torch.save({"X": X_train, "y": y_train}, os.path.join(FEATURES_DIR, "train.pt"))
torch.save({"X": X_val,   "y": y_val},   os.path.join(FEATURES_DIR, "val.pt"))
torch.save({"X": X_test,  "y": y_test},  os.path.join(FEATURES_DIR, "test.pt"))
print("Tensores guardados.")