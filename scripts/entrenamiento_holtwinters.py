"""
entrenamiento_holtwinters.py
SageMaker Processing Job — Entrenamiento Holt-Winters Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Inputs  (ProcessingInput):
    /opt/ml/processing/input/curated/  ← data/curated/embalse_guatape/

Outputs (ProcessingOutput):
    /opt/ml/processing/output/model/   → models/holtwinters/latest/
                                         models/holtwinters/versions/<fecha>/

Estructura S3:
    models/holtwinters/
    ├── latest/
    │   ├── modelo_holtwinters.pkl
    │   ├── metricas.json
    │   └── metadata.json
    └── versions/2026-06-03/
        ├── modelo_holtwinters.pkl
        ├── metricas.json
        └── metadata.json

Lógica (idéntica al notebook, cell 30):
    - ETS aditivo: trend='add', seasonal='add', seasonal_periods=365
    - Serie en nivel original (m³), sin transformar
    - IC aproximado: ±1.96 × σ residuos in-sample
"""

import subprocess, sys
for pkg in ["pyarrow"]:
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("/opt/ml/processing/input/curated")
OUTPUT_DIR = Path("/opt/ml/processing/output/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
LATEST_PREFIX   = "models/holtwinters/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"models/holtwinters/versions/{fecha_hoy}"

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/6] Cargando datos curated...")
parquet_files = list(INPUT_DIR.rglob("*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.set_index("fecha").sort_index().asfreq("D")
serie = df["volumen_m3"].dropna()

print(f"  Período : {serie.index.min().date()} → {serie.index.max().date()}")
print(f"  N       : {len(serie)} observaciones")
print(f"  Min/Max : {serie.min():,.0f} / {serie.max():,.0f} m³")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT 80/20
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/6] Split train/test 80/20...")
split_idx = int(len(serie) * 0.80)
train     = serie.iloc[:split_idx]
test      = serie.iloc[split_idx:]
print(f"  Train : {train.index.min().date()} → {train.index.max().date()} ({len(train)} obs)")
print(f"  Test  : {test.index.min().date()} → {test.index.max().date()}  ({len(test)} obs)")

# Holt-Winters necesita al menos 2 ciclos completos para inicializar seasonal_periods=365
# Con 80% de ~4168 obs = 3334 obs → 9+ años → OK
print(f"  Ciclos completos en train: {len(train) / 365:.1f} (mínimo: 2)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. AJUSTAR SOBRE TRAIN
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/6] Ajustando Holt-Winters (ETS aditivo, seasonal_periods=365)...")

hw_train = ExponentialSmoothing(
    train,
    trend="add",
    seasonal="add",
    seasonal_periods=365,
    initialization_method="estimated",
).fit(optimized=True, use_brute=False)

print(f"  Alfa  (nivel)         : {hw_train.params['smoothing_level']:.4f}")
print(f"  Beta  (tendencia)     : {hw_train.params['smoothing_trend']:.4f}")
print(f"  Gamma (estacionalidad): {hw_train.params['smoothing_seasonal']:.4f}")
print(f"  AIC : {hw_train.aic:.2f}")
print(f"  BIC : {hw_train.bic:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. MÉTRICAS IN-SAMPLE Y OUT-OF-SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/6] Métricas...")

hw_fitted = hw_train.fittedvalues
hw_resid  = train - hw_fitted
sigma_hw  = float(hw_resid.std())

mae_train  = float(mean_absolute_error(train, hw_fitted))
rmse_train = float(np.sqrt(mean_squared_error(train, hw_fitted)))
mape_train = float((np.abs((train - hw_fitted) / train) * 100).mean())

# Out-of-sample
pred_test  = hw_train.forecast(len(test))
pred_test.index = test.index
mae_test   = float(mean_absolute_error(test, pred_test))
rmse_test  = float(np.sqrt(mean_squared_error(test, pred_test)))
mape_test  = float((np.abs((test - pred_test) / test) * 100).mean())

print(f"  In-sample  → MAE: {mae_train:>15,.0f} m³  RMSE: {rmse_train:>15,.0f} m³  MAPE: {mape_train:.2f}%")
print(f"  Out-sample → MAE: {mae_test:>15,.0f} m³  RMSE: {rmse_test:>15,.0f} m³  MAPE: {mape_test:.2f}%")
print(f"  Sigma residuos (IC base): {sigma_hw:,.0f} m³")

# ══════════════════════════════════════════════════════════════════════════════
# 5. REENTRENAR CON SERIE COMPLETA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/6] Reentrenando con serie completa para producción...")

hw_final = ExponentialSmoothing(
    serie,
    trend="add",
    seasonal="add",
    seasonal_periods=365,
    initialization_method="estimated",
).fit(optimized=True, use_brute=False)

hw_resid_full = serie - hw_final.fittedvalues
sigma_final   = float(hw_resid_full.std())

print(f"  AIC final : {hw_final.aic:.2f}")
print(f"  Sigma final: {sigma_final:,.0f} m³")

# ══════════════════════════════════════════════════════════════════════════════
# 6. GUARDAR ARTEFACTOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/6] Guardando artefactos...")

# modelo_holtwinters.pkl — incluye el sigma para IC en inferencia
artefacto_model = {
    "model": hw_final,
    "sigma": sigma_final,
}
pkl_path = OUTPUT_DIR / "modelo_holtwinters.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(artefacto_model, f)

# metricas.json
metricas = {
    "mae_train" : round(mae_train,  2),
    "rmse_train": round(rmse_train, 2),
    "mape_train": round(mape_train, 4),
    "mae_test"  : round(mae_test,   2),
    "rmse_test" : round(rmse_test,  2),
    "mape_test" : round(mape_test,  4),
    "sigma_resid": round(sigma_final, 2),
}
met_path = OUTPUT_DIR / "metricas.json"
met_path.write_text(json.dumps(metricas, indent=2))

# metadata.json
metadata = {
    "embalse"             : "GUATAPE",
    "modelo"              : "Holt-Winters ETS",
    "trend"               : "add",
    "seasonal"            : "add",
    "seasonal_periods"    : 365,
    "alpha"               : round(hw_final.params["smoothing_level"],    4),
    "beta"                : round(hw_final.params["smoothing_trend"],     4),
    "gamma"               : round(hw_final.params["smoothing_seasonal"],  4),
    "aic"                 : round(hw_final.aic, 2),
    "bic"                 : round(hw_final.bic, 2),
    "metrica"             : "VoluUtilDiarMasa",
    "unidad"              : "m3",
    "fecha_entrenamiento" : datetime.now(timezone.utc).isoformat(),
    "train_start"         : str(train.index.min().date()),
    "train_end"           : str(serie.index.max().date()),
    "n_obs"               : int(len(serie)),
}
meta_path = OUTPUT_DIR / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))

# Subir a latest/ y versions/
for artefacto in [pkl_path, met_path, meta_path]:
    for prefix in [LATEST_PREFIX, VERSIONS_PREFIX]:
        s3_upload(artefacto, f"{prefix}/{artefacto.name}")

print(f"\n✅ Entrenamiento Holt-Winters completado.")
print(f"   latest/   → s3://{BUCKET}/{LATEST_PREFIX}/")
print(f"   versions/ → s3://{BUCKET}/{VERSIONS_PREFIX}/")
