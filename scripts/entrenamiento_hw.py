"""
entrenamiento_hw.py
SageMaker Processing Job — Entrenamiento Holt-Winters Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Inputs  (ProcessingInput):
    /opt/ml/processing/input/curated/  ← data/curated/embalse_guatape/

Outputs (ProcessingOutput):
    /opt/ml/processing/output/model/   → models/hw/latest/
                                         models/hw/versions/<fecha>/
"""

import subprocess, sys
for pkg in ["pyarrow", "statsmodels", "scikit-learn", "s3fs"]:
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

INPUT_DIR  = Path("/opt/ml/processing/input/curated/embalse_guatape/")
OUTPUT_DIR = Path("/opt/ml/processing/output/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
LATEST_PREFIX   = "models/hw/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"models/hw/versions/{fecha_hoy}"

PERIODO     = 365
TENDENCIA   = "add"
N_BOOTSTRAP = 500

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/7] Cargando datos curated...")

parquet_files = list(INPUT_DIR.rglob("*.parquet"))
print(f"  Parquets encontrados: {len(parquet_files)}")

if parquet_files:
    df = pd.concat([pd.read_parquet(p) for p in parquet_files], ignore_index=True)
else:
    import s3fs
    fs = s3fs.S3FileSystem()
    with fs.open(f"s3://{BUCKET}/data/curated/embalse_guatape/volumen_curated.parquet") as f:
        df = pd.read_parquet(f)

df["fecha"] = pd.to_datetime(df["fecha"])
df = df.set_index("fecha").sort_index().asfreq("D")
serie = df["volumen_m3"].dropna()

print(f"  Período : {serie.index.min().date()} → {serie.index.max().date()}")
print(f"  N       : {len(serie)} observaciones")
print(f"  Min/Max : {serie.min():,.0f} / {serie.max():,.0f} m³")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT 80/20
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] Split train/test 80/20...")
split_idx = int(len(serie) * 0.80)
train     = serie.iloc[:split_idx]
test      = serie.iloc[split_idx:]
print(f"  Train : {train.index.min().date()} → {train.index.max().date()} ({len(train)} obs)")
print(f"  Test  : {test.index.min().date()} → {test.index.max().date()}  ({len(test)} obs)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. SELECCIÓN VARIANTE: aditivo vs multiplicativo (menor AIC)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Seleccionando variante (aditivo vs multiplicativo)...")

tiene_no_positivos = (train <= 0).any()
variantes_a_probar = ["add"]
if not tiene_no_positivos:
    variantes_a_probar.append("mul")

resultados_aic = {}
for variante in variantes_a_probar:
    try:
        m = ExponentialSmoothing(
            train,
            trend=TENDENCIA,
            seasonal=variante,
            seasonal_periods=PERIODO,
            initialization_method="estimated",
            use_boxcox=False,
        ).fit(optimized=True, remove_bias=False)
        resultados_aic[variante] = m.aic
        print(f"  {variante:>3s} → AIC = {m.aic:,.2f}")
    except Exception as e:
        print(f"  {variante:>3s} → falló: {e}")

ESTACIONAL = min(resultados_aic, key=resultados_aic.get)
print(f"\n  ✔ Variante elegida: {ESTACIONAL}  (AIC={resultados_aic[ESTACIONAL]:,.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# 4. AJUSTE SOBRE TRAIN
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[4/7] Ajustando Holt-Winters ({TENDENCIA}/{ESTACIONAL}, T={PERIODO})...")

modelo_train = ExponentialSmoothing(
    train,
    trend=TENDENCIA,
    seasonal=ESTACIONAL,
    seasonal_periods=PERIODO,
    initialization_method="estimated",
    use_boxcox=False,
).fit(optimized=True, remove_bias=False)

params = modelo_train.params
print(f"  α: {params['smoothing_level']:.6f}")
print(f"  β: {params['smoothing_trend']:.6f}")
print(f"  γ: {params['smoothing_seasonal']:.6f}")
print(f"  AIC: {modelo_train.aic:,.2f}  BIC: {modelo_train.bic:,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. MÉTRICAS
#    In-sample: valores ajustados vs train
#    OOS: forecast 30 días desde fin de train (horizonte operativo)
#         NO 835 pasos — Holt-Winters multiplicativo explota en multi-step largo
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Métricas...")

def evaluar(y_true, y_pred, label):
    y_pred = np.array(y_pred)
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float((np.abs((np.array(y_true) - y_pred) / np.array(y_true)) * 100).mean())
    print(f"  {label:15s} → MAE: {mae:>12,.0f} m³  RMSE: {rmse:>12,.0f} m³  MAPE: {mape:.2f}%")
    return mae, rmse, mape

# In-sample
fitted_train = modelo_train.fittedvalues
mae_train, rmse_train, mape_train = evaluar(train, fitted_train, "In-sample")

# OOS — primeros 30 días del test
H_EVAL    = min(30, len(test))
test_eval = test.iloc[:H_EVAL]
pred_oos  = modelo_train.forecast(H_EVAL).values
mae_test, rmse_test, mape_test = evaluar(test_eval, pred_oos, f"OOS ({H_EVAL}d)")
print(f"  (OOS evaluado sobre primeros {H_EVAL} días del test — horizonte operativo)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. REENTRENAR CON SERIE COMPLETA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Reentrenando con serie completa para producción...")

modelo_final = ExponentialSmoothing(
    serie,
    trend=TENDENCIA,
    seasonal=ESTACIONAL,
    seasonal_periods=PERIODO,
    initialization_method="estimated",
    use_boxcox=False,
).fit(optimized=True, remove_bias=False)

params_final = modelo_final.params
print(f"  α: {params_final['smoothing_level']:.6f}")
print(f"  β: {params_final['smoothing_trend']:.6f}")
print(f"  γ: {params_final['smoothing_seasonal']:.6f}")
print(f"  AIC: {modelo_final.aic:,.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# 7. GUARDAR ARTEFACTOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Guardando artefactos...")

pkl_path = OUTPUT_DIR / "modelo_hw.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(modelo_final, f)

metricas = {
    "mae_train"          : round(mae_train,  2),
    "rmse_train"         : round(rmse_train, 2),
    "mape_train"         : round(mape_train, 4),
    "mae_test"           : round(mae_test,   2),
    "rmse_test"          : round(rmse_test,  2),
    "mape_test"          : round(mape_test,  4),
    "oos_horizonte_dias" : H_EVAL,
    "nota_oos"           : f"OOS sobre primeros {H_EVAL}d del test (forecast desde fin train)",
    "aic"                : round(float(modelo_final.aic),  4),
    "bic"                : round(float(modelo_final.bic),  4),
    "aicc"               : round(float(modelo_final.aicc), 4),
}
met_path = OUTPUT_DIR / "metricas.json"
met_path.write_text(json.dumps(metricas, indent=2))

metadata = {
    "embalse"             : "GUATAPE",
    "metrica"             : "VoluUtilDiarMasa",
    "unidad"              : "m3",
    "modelo"              : {
        "tipo"           : "Holt-Winters",
        "tendencia"      : TENDENCIA,
        "estacional"     : ESTACIONAL,
        "periodo"        : PERIODO,
        "n_bootstrap_ic" : N_BOOTSTRAP,
        "alpha"          : round(float(params_final["smoothing_level"]),    8),
        "beta"           : round(float(params_final["smoothing_trend"]),     8),
        "gamma"          : round(float(params_final["smoothing_seasonal"]), 8),
    },
    "fecha_entrenamiento" : datetime.now(timezone.utc).isoformat(),
    "train_start"         : str(train.index.min().date()),
    "train_end"           : str(train.index.max().date()),   # ← corte 80/20
    "serie_end"           : str(serie.index.max().date()),   # ← base forecast prod
    "n_obs_train"         : int(len(train)),
    "n_obs_total"         : int(len(serie)),
}
meta_path = OUTPUT_DIR / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))

for artefacto in [pkl_path, met_path, meta_path]:
    for prefix in [LATEST_PREFIX, VERSIONS_PREFIX]:
        s3_upload(artefacto, f"{prefix}/{artefacto.name}")

print(f"\n✅ Entrenamiento Holt-Winters completado.")
print(f"   train_end : {metadata['train_end']}  (base métricas OOS)")
print(f"   serie_end : {metadata['serie_end']}  (base forecast producción)")
print(f"   Variante  : tendencia={TENDENCIA}, estacional={ESTACIONAL}, T={PERIODO}")
