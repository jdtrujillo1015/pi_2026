"""
inferencia_hw.py
SageMaker Processing Job — Inferencia Holt-Winters Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Inputs  (ProcessingInput):
    /opt/ml/processing/input/model/    ← models/hw/latest/

Outputs (ProcessingOutput):
    /opt/ml/processing/output/preds/   → predictions/hw/latest/
                                         predictions/hw/versions/<fecha>/

Estructura de salida en S3:
    predictions/hw/
    ├── latest/
    │   ├── forecast_7d.parquet
    │   ├── forecast_15d.parquet
    │   └── forecast_30d.parquet
    └── versions/
        └── 2026-06-03/
            ├── forecast_7d.parquet
            ├── forecast_15d.parquet
            └── forecast_30d.parquet

Columnas de cada parquet:
    fecha | volumen_predicho_m3 | ic_lower_95 | ic_upper_95

IC 95 %
───────
Se usa bootstrap sobre residuos in-sample (n=N_BOOTSTRAP trayectorias).
En cada trayectoria bootstrap se remuestrea con reemplazo el vector de
residuos del modelo, se añade al forecast puntual y se calculan los
percentiles 2.5 y 97.5.  Este enfoque es consistente con el MC-Dropout
usado en el LSTM y no requiere supuesto de normalidad.
"""

import subprocess, sys
for pkg in ["pyarrow", "statsmodels", "scikit-learn"]:
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

# ── Rutas ─────────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("/opt/ml/processing/input/model")
OUTPUT_DIR = Path("/opt/ml/processing/output/preds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
LATEST_PREFIX   = "predictions/hw/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"predictions/hw/versions/{fecha_hoy}"

HORIZONTES   = [7, 15, 30]
N_BOOTSTRAP  = 500        # trayectorias para IC bootstrap

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR MODELO Y METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Cargando modelo y metadata...")

with open(INPUT_DIR / "modelo_hw.pkl", "rb") as f:
    modelo_hw = pickle.load(f)

metadata = json.loads((INPUT_DIR / "metadata.json").read_text())
metricas = json.loads((INPUT_DIR / "metricas.json").read_text())

cfg = metadata["modelo"]

print(f"  Modelo            : Holt-Winters tendencia={cfg['tendencia']}, estacional={cfg['estacional']}")
print(f"  Período estacional: {cfg['periodo']} días")
print(f"  α / β / γ         : {cfg['alpha']} / {cfg['beta']} / {cfg['gamma']}")
print(f"  Entrenado el      : {metadata['fecha_entrenamiento']}")
print(f"  Datos hasta       : {metadata['serie_end']}")
print(f"  MAE test          : {metricas['mae_test']:,.0f} m³")
print(f"  MAPE test         : {metricas['mape_test']:.2f}%")
print(f"  AIC               : {metricas['aic']:,.2f}")

fecha_base = pd.to_datetime(metadata["serie_end"]) + timedelta(days=1)
print(f"  Predicciones desde: {fecha_base.date()}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. RECUPERAR RESIDUOS DESDE S3 PARA EL BOOTSTRAP
#    Los residuos se recalculan sobre la serie curated completa usando el
#    modelo entrenado, de modo que el IC refleje la dispersión real.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/4] Recuperando serie curated para calcular residuos bootstrap...")

CURATED_PREFIX = "data/curated/embalse_guatape"

paginator = s3.get_paginator("list_objects_v2")
pages     = paginator.paginate(Bucket=BUCKET, Prefix=CURATED_PREFIX)
s3_keys   = [obj["Key"] for page in pages for obj in page.get("Contents", [])
             if obj["Key"].endswith(".parquet")]

if not s3_keys:
    raise FileNotFoundError(
        f"No se encontraron parquets bajo s3://{BUCKET}/{CURATED_PREFIX}"
    )

frames = []
for key in s3_keys:
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    buf = obj["Body"].read()
    frames.append(pd.read_parquet(__import__("io").BytesIO(buf)))

df_curated = pd.concat(frames, ignore_index=True)
df_curated["fecha"] = pd.to_datetime(df_curated["fecha"])
df_curated = df_curated.set_index("fecha").sort_index().asfreq("D")
serie      = df_curated["volumen_m3"].dropna()

# Residuos = real − ajustado (in-sample del modelo)
residuos = (serie - modelo_hw.fittedvalues).dropna().values

print(f"  Serie cubierta    : {serie.index.min().date()} → {serie.index.max().date()}")
print(f"  Residuos          : n={len(residuos)}  std={residuos.std():,.0f} m³")


# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERAR FORECASTS CON IC BOOTSTRAP
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Generando forecasts...")

def forecast_con_bootstrap(pasos: int, n_bootstrap: int = N_BOOTSTRAP):
    """
    Genera 'pasos' predicciones futuras con IC 95 % por bootstrap.

    Algoritmo
    ---------
    1. Forecast puntual h-step del modelo ajustado.
    2. Para cada trayectoria bootstrap:
       - Remuestrear 'pasos' residuos con reemplazo.
       - Añadirlos al forecast puntual (supuesto: errores aditivos e i.i.d.).
    3. Percentiles 2.5 y 97.5 sobre las n_bootstrap trayectorias.

    Parámetros
    ----------
    pasos       : horizonte en días
    n_bootstrap : número de trayectorias

    Retorna
    -------
    mean_pred : array (pasos,)
    lower_95  : array (pasos,)
    upper_95  : array (pasos,)
    """
    # Forecast puntual (h-step ahead)
    fc_puntual = modelo_hw.forecast(pasos).values   # shape (pasos,)

    # Bootstrap: remuestreo de residuos
    rng = np.random.default_rng(seed=42)
    trayectorias = np.empty((n_bootstrap, pasos))

    for i in range(n_bootstrap):
        muestra = rng.choice(residuos, size=pasos, replace=True)
        trayectorias[i] = fc_puntual + muestra

    lower_95 = np.percentile(trayectorias,  2.5, axis=0)
    upper_95 = np.percentile(trayectorias, 97.5, axis=0)

    return fc_puntual, lower_95, upper_95


archivos_generados = []

for h in HORIZONTES:
    fechas = pd.date_range(start=fecha_base, periods=h, freq="D")
    valores, ic_lower, ic_upper = forecast_con_bootstrap(pasos=h)

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
    print(f"    IC 95%: [{ic_lower.mean():>15,.0f} , {ic_upper.mean():>15,.0f}] m³ (media)")


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
print("\n✅ Inferencia Holt-Winters completada.")
