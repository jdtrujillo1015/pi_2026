"""
entrenamiento_lstm.py
SageMaker Processing Job — Entrenamiento LSTM Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Inputs  (ProcessingInput):
    /opt/ml/processing/input/curated/  ← data/curated/embalse_guatape/

Outputs (ProcessingOutput):
    /opt/ml/processing/output/model/   → models/lstm/latest/
                                         models/lstm/versions/<fecha>/

Estructura de salida en S3:
    models/lstm/
    ├── latest/
    │   ├── modelo_lstm.keras
    │   ├── scaler.pkl
    │   ├── metricas.json
    │   └── metadata.json
    └── versions/
        └── 2026-06-03/
            ├── modelo_lstm.keras
            ├── scaler.pkl
            ├── metricas.json
            └── metadata.json
"""

import subprocess, sys
for pkg in ["pyarrow", "tensorflow", "scikit-learn"]:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

import json, pickle, warnings
from datetime import datetime, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── Rutas ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("/opt/ml/processing/input/curated/embalse_guatape/")
OUTPUT_DIR = Path("/opt/ml/processing/output/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
LATEST_PREFIX   = "models/lstm/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"models/lstm/versions/{fecha_hoy}"

# ── Hiperparámetros ────────────────────────────────────────────────────────────
VENTANA      = 30        # días de historia que ve el modelo (look-back)
BATCH_SIZE   = 32
EPOCHS       = 200
LSTM_UNITS   = [64, 32]  # capas LSTM apiladas
DROPOUT_RATE = 0.2
PATIENCE     = 20        # early stopping

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/8] Cargando datos curated...")
parquet_files = list(INPUT_DIR.rglob("*.parquet"))
print(f"  Archivos: {[p.name for p in parquet_files]}")

df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.set_index("fecha").sort_index().asfreq("D")
serie = df["volumen_m3"].dropna()

print(f"  Período : {serie.index.min().date()} → {serie.index.max().date()}")
print(f"  N       : {len(serie)} observaciones")
print(f"  Min/Max : {serie.min():,.0f} / {serie.max():,.0f} m³")
print(f"  Media   : {serie.mean():,.0f} m³")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT 80/20 CRONOLÓGICO
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/8] Split train/test 80/20...")
split_idx = int(len(serie) * 0.80)
train     = serie.iloc[:split_idx]
test      = serie.iloc[split_idx:]
print(f"  Train : {train.index.min().date()} → {train.index.max().date()} ({len(train)} obs)")
print(f"  Test  : {test.index.min().date()} → {test.index.max().date()}  ({len(test)} obs)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ESCALADO (MinMaxScaler sobre train)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] Escalando serie...")
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
# Para evaluación OOS se transforma la serie completa y se extrae la parte test
serie_scaled  = scaler.transform(serie.values.reshape(-1, 1))
test_scaled   = serie_scaled[split_idx:]
print(f"  Rango original  : [{serie.min():.0f}, {serie.max():.0f}] m³")
print(f"  Rango escalado  : [{train_scaled.min():.4f}, {train_scaled.max():.4f}]")

# ══════════════════════════════════════════════════════════════════════════════
# 4. CREAR SECUENCIAS (look-back window)
# ══════════════════════════════════════════════════════════════════════════════
def crear_secuencias(data: np.ndarray, ventana: int):
    """Convierte un array 1-D escalado en pares (X, y) con shape (N, ventana, 1)."""
    X, y = [], []
    for i in range(ventana, len(data)):
        X.append(data[i - ventana:i, 0])
        y.append(data[i, 0])
    return np.array(X)[..., np.newaxis], np.array(y)

print(f"\n[4/8] Construyendo secuencias (ventana={VENTANA} días)...")
# Para train: secuencias sobre datos de entrenamiento
X_train, y_train = crear_secuencias(train_scaled, VENTANA)

# Para test: necesitamos los últimos VENTANA valores de train como contexto
serie_eval  = serie_scaled[split_idx - VENTANA:]
X_test, y_test = crear_secuencias(serie_eval, VENTANA)

print(f"  X_train shape : {X_train.shape}  →  {X_train.shape[0]} muestras")
print(f"  X_test  shape : {X_test.shape}   →  {X_test.shape[0]} muestras")

# ══════════════════════════════════════════════════════════════════════════════
# 5. ARQUITECTURA LSTM
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] Construyendo modelo LSTM...")

model = Sequential(name="lstm_guatape")

for i, units in enumerate(LSTM_UNITS):
    return_seq = (i < len(LSTM_UNITS) - 1)   # True en todas salvo la última capa
    if i == 0:
        model.add(LSTM(units, return_sequences=return_seq,
                       input_shape=(VENTANA, 1), name=f"lstm_{i+1}"))
    else:
        model.add(LSTM(units, return_sequences=return_seq, name=f"lstm_{i+1}"))
    model.add(Dropout(DROPOUT_RATE, name=f"dropout_{i+1}"))

model.add(Dense(1, name="output"))

model.compile(optimizer="adam", loss="mean_squared_error")
model.summary()

# ══════════════════════════════════════════════════════════════════════════════
# 6. ENTRENAMIENTO CON EARLY STOPPING
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[6/8] Entrenando (max {EPOCHS} epochs, patience={PATIENCE})...")

callbacks = [
    EarlyStopping(monitor="val_loss", patience=PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10,
                      min_lr=1e-6, verbose=1),
]

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    callbacks=callbacks,
    verbose=1,
    shuffle=False,         # series temporales: NO mezclar
)

epocas_reales = len(history.history["loss"])
loss_final    = history.history["val_loss"][-1]
print(f"\n  Épocas ejecutadas : {epocas_reales}")
print(f"  Val-loss final    : {loss_final:.6f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. MÉTRICAS IN-SAMPLE Y OUT-OF-SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Calculando métricas...")

def evaluar(X, y_true_scaled, label: str):
    y_pred_scaled = model.predict(X, verbose=0)
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
    y_true = scaler.inverse_transform(y_true_scaled.reshape(-1, 1)).flatten()
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float((np.abs((y_true - y_pred) / y_true) * 100).mean())
    print(f"  {label:12s} → MAE: {mae:>15,.0f} m³  RMSE: {rmse:>15,.0f} m³  MAPE: {mape:.2f}%")
    return mae, rmse, mape, y_pred

mae_train, rmse_train, mape_train, _ = evaluar(X_train, y_train, "In-sample")
mae_test,  rmse_test,  mape_test, _  = evaluar(X_test,  y_test,  "Out-sample")

# ══════════════════════════════════════════════════════════════════════════════
# 8. GUARDAR ARTEFACTOS  →  output local  +  versión en S3
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Guardando artefactos...")

# ── modelo_lstm.keras ─────────────────────────────────────────────────────────
modelo_path = OUTPUT_DIR / "modelo_lstm.keras"
model.save(str(modelo_path))

# ── scaler.pkl (necesario para invertir la normalización en inferencia) ────────
scaler_path = OUTPUT_DIR / "scaler.pkl"
with open(scaler_path, "wb") as f:
    pickle.dump(scaler, f)

# ── metricas.json ─────────────────────────────────────────────────────────────
metricas = {
    "mae_train" : round(mae_train,  2),
    "rmse_train": round(rmse_train, 2),
    "mape_train": round(mape_train, 4),
    "mae_test"  : round(mae_test,   2),
    "rmse_test" : round(rmse_test,  2),
    "mape_test" : round(mape_test,  4),
}
met_path = OUTPUT_DIR / "metricas.json"
met_path.write_text(json.dumps(metricas, indent=2))

# ── metadata.json ─────────────────────────────────────────────────────────────
metadata = {
    "embalse"            : "GUATAPE",
    "metrica"            : "VoluUtilDiarMasa",
    "unidad"             : "m3",
    "arquitectura"       : {
        "tipo"          : "LSTM",
        "capas_lstm"    : LSTM_UNITS,
        "dropout"       : DROPOUT_RATE,
        "ventana_dias"  : VENTANA,
        "batch_size"    : BATCH_SIZE,
        "epocas_max"    : EPOCHS,
        "epocas_reales" : epocas_reales,
        "optimizer"     : "adam",
        "loss"          : "mse",
    },
    "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
    "train_start"        : str(train.index.min().date()),
    "train_end"          : str(serie.index.max().date()),
    "n_obs"              : int(len(serie)),
}
meta_path = OUTPUT_DIR / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))

# ── Subir a latest/ y a versions/<fecha>/ ────────────────────────────────────
artefactos = [modelo_path, scaler_path, met_path, meta_path]

for artefacto in artefactos:
    for prefix in [LATEST_PREFIX, VERSIONS_PREFIX]:
        s3_upload(artefacto, f"{prefix}/{artefacto.name}")

print(f"\n✅ Entrenamiento LSTM completado.")
print(f"   latest/   → s3://{BUCKET}/{LATEST_PREFIX}/")
print(f"   versions/ → s3://{BUCKET}/{VERSIONS_PREFIX}/")
