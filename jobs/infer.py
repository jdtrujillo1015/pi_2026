import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from datetime import datetime
from config import CURATED_DIR, PREDICTIONS_DIR, ARTIFACTS_DIR, SCALER_PATH, MODEL_PATH
from config import WINDOW_SIZE, HORIZON, HIDDEN_SIZE, NUM_LAYERS

# Arquitectura (debe ser idéntica a train.py)
class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=HIDDEN_SIZE,
                            num_layers=NUM_LAYERS, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(HIDDEN_SIZE, HORIZON)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# Cargar modelo y scaler
scaler = joblib.load(SCALER_PATH)
model  = LSTMForecaster()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Leer curated y tomar los últimos WINDOW_SIZE días
df = pd.read_csv(os.path.join(CURATED_DIR, "volumen_m3.csv"))
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

ultimos = df["volumen_m3"].values[-WINDOW_SIZE:].reshape(-1, 1)
ultima_fecha = df["fecha"].iloc[-1]

# Escalar
ultimos_s = scaler.transform(ultimos)
x = torch.tensor(ultimos_s, dtype=torch.float32).unsqueeze(0)  # (1, 90, 1)

# Predecir
with torch.no_grad():
    pred_s = model(x).numpy()

# Desescalar
pred = scaler.inverse_transform(pred_s).flatten()

# Construir dataframe de predicciones
fechas_futuras = pd.date_range(
    start=ultima_fecha + pd.Timedelta(days=1),
    periods=HORIZON,
    freq="D"
)

resultado = pd.DataFrame({
    "fecha": fechas_futuras,
    "volumen_m3_predicho": pred
})

# Guardar
fecha_hoy = datetime.today().strftime("%Y-%m-%d")
ruta = os.path.join(PREDICTIONS_DIR, f"prediccion_{fecha_hoy}.csv")
resultado.to_csv(ruta, index=False, encoding="utf-8")
print(f"Predicción guardada: {ruta}")
print(resultado)