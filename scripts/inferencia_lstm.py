"""
inferencia_lstm.py
SageMaker Processing Job — Inferencia LSTM Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Inputs  (ProcessingInput):
    /opt/ml/processing/input/model/    ← models/lstm/latest/

Outputs (ProcessingOutput):
    /opt/ml/processing/output/preds/   → predictions/lstm/latest/
                                         predictions/lstm/versions/<fecha>/

Estructura de salida en S3:
    predictions/lstm/
    ├── latest/
    │   ├── forecast_7d.parquet
    │   ├── forecast_15d.parquet
    │   └── forecast_30d.parquet
    └── versions/
        └── 2026-06-03/
            ├── forecast_7d.parquet
            ├── forecast_15d.parquet
            └── forecast_30d.parquet
"""

import subprocess, sys
for pkg in ["pyarrow", "tensorflow", "scikit-learn"]:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg, "-q"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

import json, pickle, warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ── Rutas ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("/opt/ml/processing/input/model")
OUTPUT_DIR = Path("/opt/ml/processing/output/preds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
LATEST_PREFIX   = "predictions/lstm/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"predictions/lstm/versions/{fecha_hoy}"

HORIZONTES = [7, 15, 30]

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR MODELO, SCALER Y METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Cargando modelo y metadata...")

model = keras.models.load_model(str(INPUT_DIR / "modelo_lstm.keras"))

with open(INPUT_DIR / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

metadata = json.loads((INPUT_DIR / "metadata.json").read_text())
metricas = json.loads((INPUT_DIR / "metricas.json").read_text())

arq      = metadata["arquitectura"]
VENTANA  = arq["ventana_dias"]

print(f"  Modelo            : LSTM {arq['capas_lstm']} units, dropout={arq['dropout']}")
print(f"  Ventana look-back : {VENTANA} días")
print(f"  Entrenado el      : {metadata['fecha_entrenamiento']}")
print(f"  Datos hasta       : {metadata['train_end']}")
print(f"  MAE test          : {metricas['mae_test']:,.0f} m³")
print(f"  MAPE test         : {metricas['mape_test']:.2f}%")

fecha_base = pd.to_datetime(metadata["train_end"]) + timedelta(days=1)
print(f"  Predicciones desde: {fecha_base.date()}")

model.summary()


# ══════════════════════════════════════════════════════════════════════════════
# 2. RECUPERAR LA VENTANA DE CONTEXTO DESDE S3
#    (últimos VENTANA días de la serie de entrenamiento)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Recuperando ventana de contexto desde S3...")

# Los datos curated se leen para obtener el contexto inicial del forecast.
# Se usa el mismo bucket y prefijo que el job de entrenamiento.
CURATED_PREFIX = "data/curated/embalse_guatape"

paginator  = s3.get_paginator("list_objects_v2")
pages      = paginator.paginate(Bucket=BUCKET, Prefix=CURATED_PREFIX)
s3_keys    = [obj["Key"] for page in pages for obj in page.get("Contents", [])
              if obj["Key"].endswith(".parquet")]

if not s3_keys:
    raise FileNotFoundError(
        f"No se encontraron parquets bajo s3://{BUCKET}/{CURATED_PREFIX}"
    )

frames = []
for key in s3_keys:
    obj  = s3.get_object(Bucket=BUCKET, Key=key)
    buf  = obj["Body"].read()
    frames.append(pd.read_parquet(__import__("io").BytesIO(buf)))

df_curated = pd.concat(frames, ignore_index=True)
df_curated["fecha"] = pd.to_datetime(df_curated["fecha"])
df_curated = df_curated.set_index("fecha").sort_index().asfreq("D")
serie      = df_curated["volumen_m3"].dropna()

# Los últimos VENTANA valores escalados sirven como semilla del forecast
ultimos_vals = scaler.transform(serie.values[-VENTANA:].reshape(-1, 1))
print(f"  Serie cubierta    : {serie.index.min().date()} → {serie.index.max().date()}")
print(f"  Ventana extraída  : {serie.index[-VENTANA].date()} → {serie.index[-1].date()}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERAR FORECASTS (predicción iterativa paso a paso)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Generando forecasts...")

def forecast_iterativo(seed: np.ndarray, pasos: int, n_bootstrap: int = 200):
    """
    Genera 'pasos' predicciones futuras de forma recursiva (1-step ahead).

    Para estimar incertidumbre se utiliza Monte Carlo Dropout:
    se ejecutan 'n_bootstrap' pasadas hacia adelante con dropout activo
    y se calculan percentiles 2.5 y 97.5 como intervalo de confianza al 95 %.

    Parámetros
    ----------
    seed        : array (VENTANA, 1) escalado — valores iniciales
    pasos       : horizonte en días
    n_bootstrap : número de muestras MC-Dropout

    Retorna
    -------
    mean_pred   : array (pasos,) — predicción media (sin escalar)
    lower_95    : array (pasos,) — percentil  2.5 (sin escalar)
    upper_95    : array (pasos,) — percentil 97.5 (sin escalar)
    """
    # ── n_bootstrap trayectorias con dropout activo ──────────────────────────
    all_preds = []  # shape final: (n_bootstrap, pasos)

    for _ in range(n_bootstrap):
        ventana  = seed.copy()          # (VENTANA, 1)
        trayecto = []
        for _ in range(pasos):
            X   = ventana[np.newaxis, :, :]          # (1, VENTANA, 1)
            yp  = model(X, training=True).numpy()[0, 0]  # dropout activo
            trayecto.append(yp)
            ventana = np.append(ventana[1:], [[yp]], axis=0)
        all_preds.append(trayecto)

    all_preds  = np.array(all_preds)                     # (n_bootstrap, pasos)
    mean_scaled = all_preds.mean(axis=0).reshape(-1, 1)
    lo_scaled   = np.percentile(all_preds,  2.5, axis=0).reshape(-1, 1)
    hi_scaled   = np.percentile(all_preds, 97.5, axis=0).reshape(-1, 1)

    mean_pred = scaler.inverse_transform(mean_scaled).flatten()
    lower_95  = scaler.inverse_transform(lo_scaled).flatten()
    upper_95  = scaler.inverse_transform(hi_scaled).flatten()
    return mean_pred, lower_95, upper_95


archivos_generados = []

for h in HORIZONTES:
    fechas = pd.date_range(start=fecha_base, periods=h, freq="D")
    valores, ic_lower, ic_upper = forecast_iterativo(ultimos_vals, pasos=h)

    df_pred = pd.DataFrame({
        "fecha"              : fechas.strftime("%Y-%m-%d"),
        "volumen_predicho_m3": np.round(valores,   2),
        "ic_lower_95"        : np.round(ic_lower,  2),
        "ic_upper_95"        : np.round(ic_upper,  2),
    })

    nombre     = f"forecast_{h}d.parquet"
    local_path = OUTPUT_DIR / nombre
    df_pred.to_parquet(local_path, index=False)
    archivos_generados.append((nombre, local_path))

    print(f"\n  Horizonte {h:2d} días → {fechas[0].date()} … {fechas[-1].date()}")
    print(f"    Media : {valores.mean():>15,.0f} m³")
    print(f"    Min   : {valores.min():>15,.0f} m³")
    print(f"    Max   : {valores.max():>15,.0f} m³")


# ══════════════════════════════════════════════════════════════════════════════
# 4. SUBIR A latest/  Y  versions/<fecha>/
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Subiendo a S3...")

for nombre, local_path in archivos_generados:
    for prefix in [LATEST_PREFIX, VERSIONS_PREFIX]:
        s3_upload(local_path, f"{prefix}/{nombre}")

print(f"\n  latest/   → s3://{BUCKET}/{LATEST_PREFIX}/")
print(f"  versions/ → s3://{BUCKET}/{VERSIONS_PREFIX}/")
for h in HORIZONTES:
    print(f"  forecast_{h}d.parquet → cols: fecha | volumen_predicho_m3 | ic_lower_95 | ic_upper_95")
print("\n✅ Inferencia LSTM completada.")
