"""
entrenamiento_arima.py
SageMaker Processing Job — Entrenamiento ARIMA Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Inputs  (ProcessingInput):
    /opt/ml/processing/input/curated/  ← data/curated/embalse_guatape/

Outputs (ProcessingOutput):
    /opt/ml/processing/output/model/   → models/arima/latest/
                                         models/arima/versions/<fecha>/
"""

import subprocess, sys
for pkg in ["pmdarima", "pyarrow", "s3fs"]:
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
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

INPUT_DIR  = Path("/opt/ml/processing/input/curated/embalse_guatape/")
OUTPUT_DIR = Path("/opt/ml/processing/output/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
LATEST_PREFIX   = "models/arima/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"models/arima/versions/{fecha_hoy}"

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/8] Cargando datos curated...")

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
print(f"  Min/Max : {serie.min():.4f} / {serie.max():.4f} m³")

# ══════════════════════════════════════════════════════════════════════════════
# 2. SPLIT 80/20
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/8] Split train/test 80/20...")
split_idx = int(len(serie) * 0.80)
train     = serie.iloc[:split_idx]
test      = serie.iloc[split_idx:]
print(f"  Train : {train.index.min().date()} → {train.index.max().date()} ({len(train)} obs)")
print(f"  Test  : {test.index.min().date()} → {test.index.max().date()}  ({len(test)} obs)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. ADF NIVEL + ADF RETORNOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] Pruebas ADF...")
adf_nivel = adfuller(train.dropna())
print(f"  ADF nivel    : stat={adf_nivel[0]:.4f}  p={adf_nivel[1]:.4f}"
      f"  → {'NO estacionaria' if adf_nivel[1] >= 0.05 else 'estacionaria'}")

retornos = np.log(train).diff().dropna()
adf_ret  = adfuller(retornos)
print(f"  ADF retornos : stat={adf_ret[0]:.4f}  p={adf_ret[1]:.4f}"
      f"  → {'NO estacionaria' if adf_ret[1] >= 0.05 else 'estacionaria'}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. ACF/PACF → p_max, q_max
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/8] Rezagos significativos ACF/PACF...")
N      = len(retornos)
umbral = 1.96 / np.sqrt(N)
acf_v  = acf(retornos, nlags=40)
pacf_v = pacf(retornos, nlags=40, method="ywm")

lags_q = [i for i in range(1, len(acf_v))  if abs(acf_v[i])  > umbral]
lags_p = [i for i in range(1, len(pacf_v)) if abs(pacf_v[i]) > umbral]

p_max = min(len(lags_p), 7)
q_max = min(len(lags_q), 5)
print(f"  Umbral : {umbral:.4f}")
print(f"  lags_p : {lags_p[:10]}  →  p_max={p_max}")
print(f"  lags_q : {lags_q[:10]}  →  q_max={q_max}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. AUTO_ARIMA (d=1 fijo)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] auto_arima (d=1, stepwise)...")
modelo_auto = auto_arima(
    train,
    start_p=0, start_q=0,
    max_p=p_max, max_q=q_max,
    d=1,
    seasonal=False,
    information_criterion="aic",
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
    stepwise=True,
)
p_opt, d_opt, q_opt = modelo_auto.order
print(f"\n  Mejor orden: ARIMA{modelo_auto.order}")

# ══════════════════════════════════════════════════════════════════════════════
# 6. AJUSTE SOBRE TRAIN + MÉTRICAS
#    OOS: forecast de 30 días desde fin de train (horizonte operativo real)
#    No 834 pasos — el ARIMA también acumula error en multi-step largo
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] Ajuste y métricas...")
modelo_eval = ARIMA(train, order=(p_opt, d_opt, q_opt)).fit()

# In-sample
fitted     = modelo_eval.fittedvalues.dropna()
y_tr       = train.loc[fitted.index]
mae_train  = float(mean_absolute_error(y_tr, fitted))
rmse_train = float(np.sqrt(mean_squared_error(y_tr, fitted)))
mape_train = float((np.abs((y_tr - fitted) / y_tr) * 100).mean())

# OOS — primeros 30 días del test (horizonte operativo: 7/15/30d)
H_EVAL    = min(30, len(test))
test_eval = test.iloc[:H_EVAL]

fc_oos       = modelo_eval.get_forecast(steps=H_EVAL)
pred_test    = fc_oos.predicted_mean
pred_test.index = test_eval.index

mae_test  = float(mean_absolute_error(test_eval, pred_test))
rmse_test = float(np.sqrt(mean_squared_error(test_eval, pred_test)))
mape_test = float((np.abs((test_eval - pred_test) / test_eval) * 100).mean())

print(f"  In-sample       → MAE: {mae_train:.4f}  RMSE: {rmse_train:.4f}  MAPE: {mape_train:.2f}%")
print(f"  OOS ({H_EVAL}d)     → MAE: {mae_test:.4f}  RMSE: {rmse_test:.4f}  MAPE: {mape_test:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 7. REENTRENAR CON SERIE COMPLETA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Reentrenando con serie completa para producción...")
modelo_final = ARIMA(serie, order=(p_opt, d_opt, q_opt)).fit()
print(modelo_final.summary())

# ══════════════════════════════════════════════════════════════════════════════
# 8. GUARDAR ARTEFACTOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Guardando artefactos...")

pkl_path = OUTPUT_DIR / "modelo_arima.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(modelo_final, f)

metricas = {
    "mae_train"          : round(mae_train,  4),
    "rmse_train"         : round(rmse_train, 4),
    "mape_train"         : round(mape_train, 4),
    "mae_test"           : round(mae_test,   4),
    "rmse_test"          : round(rmse_test,  4),
    "mape_test"          : round(mape_test,  4),
    "oos_horizonte_dias" : H_EVAL,
    "nota_oos"           : f"OOS calculado sobre primeros {H_EVAL}d del test (forecast multi-step desde fin train)",
}
met_path = OUTPUT_DIR / "metricas.json"
met_path.write_text(json.dumps(metricas, indent=2))

metadata = {
    "embalse"             : "GUATAPE",
    "metrica"             : "VoluUtilDiarMasa",
    "unidad"              : "m3",
    "order"               : list(modelo_final.model.order),
    "fecha_entrenamiento" : datetime.now(timezone.utc).isoformat(),
    "train_start"         : str(train.index.min().date()),
    "train_end"           : str(train.index.max().date()),   # ← fin del TRAIN 80%
    "serie_end"           : str(serie.index.max().date()),   # ← fin de la serie completa
    "n_obs_train"         : int(len(train)),
    "n_obs_total"         : int(len(serie)),
}
meta_path = OUTPUT_DIR / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))

for artefacto in [pkl_path, met_path, meta_path]:
    for prefix in [LATEST_PREFIX, VERSIONS_PREFIX]:
        s3_upload(artefacto, f"{prefix}/{artefacto.name}")

print(f"\n✅ Entrenamiento ARIMA completado.")
print(f"   train_end  : {metadata['train_end']}  (base métricas OOS)")
print(f"   serie_end  : {metadata['serie_end']}  (base forecast producción)")
print(f"   latest/    → s3://{BUCKET}/{LATEST_PREFIX}/")
print(f"   versions/  → s3://{BUCKET}/{VERSIONS_PREFIX}/")
