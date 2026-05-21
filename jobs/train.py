import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from config import FEATURES_DIR, ARTIFACTS_DIR, MODEL_PATH, SCALER_PATH
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, HIDDEN_SIZE, NUM_LAYERS, GRAD_CLIP, HORIZON

# Cargar tensores
train = torch.load(os.path.join(FEATURES_DIR, "train.pt"))
val   = torch.load(os.path.join(FEATURES_DIR, "val.pt"))
test  = torch.load(os.path.join(FEATURES_DIR, "test.pt"))

X_train, y_train = train["X"], train["y"]
X_val,   y_val   = val["X"],   val["y"]
X_test,  y_test  = test["X"],  test["y"]

train_loader = DataLoader(TensorDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=False)

# Modelo
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(HIDDEN_SIZE, HORIZON)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

model     = LSTMForecaster()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_epoch    = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()

    print(f"Época {epoch:03d}/{EPOCHS} · val_loss={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch    = epoch
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"  → Modelo guardado (mejor val_loss={best_val_loss:.6f})")

# Cargar mejor modelo y evaluar sobre test
print(f"\nMejor época: {best_epoch} · val_loss={best_val_loss:.6f}")
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

scaler = joblib.load(SCALER_PATH)

with torch.no_grad():
    test_pred_s = model(X_test).numpy()
    test_real_s = y_test.numpy()

# Desescalar
test_pred = scaler.inverse_transform(test_pred_s)
test_real = scaler.inverse_transform(test_real_s)

mae  = mean_absolute_error(test_real, test_pred)
rmse = np.sqrt(mean_squared_error(test_real, test_pred))
r2   = r2_score(test_real, test_pred)
mape = np.mean(np.abs((test_real - test_pred) / test_real)) * 100

print(f"\n── Métricas sobre test set ──")
print(f"MAE  : {mae:,.0f} m³")
print(f"RMSE : {rmse:,.0f} m³")
print(f"MAPE : {mape:.2f}%")
print(f"R²   : {r2:.4f}")

# Guardar métricas
metrics = {
    "mae":        float(round(mae, 2)),
    "rmse":       float(round(rmse, 2)),
    "mape":       float(round(mape, 4)),
    "r2":         float(round(r2, 4)),
    "best_epoch": int(best_epoch),
    "val_loss":   float(round(best_val_loss, 6)),
    "epochs":     int(EPOCHS)
}

ruta = os.path.join(ARTIFACTS_DIR, "metrics.json")
with open(ruta, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMétricas guardadas: {ruta}")