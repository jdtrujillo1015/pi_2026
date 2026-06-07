"""
inferencia_arima.py
SageMaker Processing Job — Inferencia ARIMA Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Inputs  (ProcessingInput):
    /opt/ml/processing/input/model/    ← models/arima/latest/

Outputs (ProcessingOutput):
    /opt/ml/processing/output/preds/   → predictions/arima/latest/
                                         predictions/arima/versions/<fecha>/
"""

import subprocess, sys
for pkg in ["pmdarima", "pyarrow"]:
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

warnings.filterwarnings("ignore")

INPUT_DIR  = Path("/opt/ml/processing/input/model")
OUTPUT_DIR = Path("/opt/ml/processing/output/preds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
LATEST_PREFIX   = "predictions/arima/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"predictions/arima/versions/{fecha_hoy}"
HORIZONTES      = [7, 15, 30]

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR MODELO Y METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Cargando modelo y metadata...")

with open(INPUT_DIR / "modelo_arima.pkl", "rb") as f:
    model = pickle.load(f)

metadata = json.loads((INPUT_DIR / "metadata.json").read_text())
metricas = json.loads((INPUT_DIR / "metricas.json").read_text())

order = tuple(metadata["order"])
print(f"  Modelo       : ARIMA{order}")
print(f"  Entrenado el : {metadata['fecha_entrenamiento']}")
print(f"  train_end    : {metadata['train_end']}  (corte 80/20 — base validación)")
print(f"  serie_end    : {metadata['serie_end']}  (último dato real — base producción)")
print(f"  MAE OOS      : {metricas['mae_test']:.4f} m³  (primeros {metricas['oos_horizonte_dias']}d del test)")
print(f"  MAPE OOS     : {metricas['mape_test']:.2f}%")

# ── Fecha base del forecast: día siguiente al último dato de la serie completa
fecha_base = pd.to_datetime(metadata["serie_end"]) + timedelta(days=1)
print(f"\n  Predicciones desde: {fecha_base.date()}  (serie_end + 1)")

# ══════════════════════════════════════════════════════════════════════════════
# 2. GENERAR FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Generando forecasts...")

archivos_generados = []

for h in HORIZONTES:
    fc_res  = model.get_forecast(steps=h)
    valores = fc_res.predicted_mean.values
    ci      = fc_res.conf_int(alpha=0.05)
    fechas  = pd.date_range(start=fecha_base, periods=h, freq="D")

    df_pred = pd.DataFrame({
        "fecha"              : fechas.strftime("%Y-%m-%d"),
        "volumen_predicho_m3": np.round(valores, 4),
        "ic_lower_95"        : np.round(ci.iloc[:, 0].values, 4),
        "ic_upper_95"        : np.round(ci.iloc[:, 1].values, 4),
    })

    nombre     = f"forecast_{h}d.parquet"
    local_path = OUTPUT_DIR / nombre
    df_pred.to_parquet(local_path, index=False)
    archivos_generados.append((nombre, local_path))

    print(f"\n  Horizonte {h:2d} días → {fechas[0].date()} … {fechas[-1].date()}")
    print(f"    Media : {valores.mean():>15,.4f} m³")
    print(f"    Min   : {valores.min():>15,.4f} m³")
    print(f"    Max   : {valores.max():>15,.4f} m³")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SUBIR A S3
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Subiendo a S3...")
for nombre, local_path in archivos_generados:
    for prefix in [LATEST_PREFIX, VERSIONS_PREFIX]:
        s3_upload(local_path, f"{prefix}/{nombre}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Resumen:")
print(f"  latest/   → s3://{BUCKET}/{LATEST_PREFIX}/")
print(f"  versions/ → s3://{BUCKET}/{VERSIONS_PREFIX}/")
for h in HORIZONTES:
    print(f"  forecast_{h}d.parquet → fecha | volumen_predicho_m3 | ic_lower_95 | ic_upper_95")
print("\n✅ Inferencia ARIMA completada.")
