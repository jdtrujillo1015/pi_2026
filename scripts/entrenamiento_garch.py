"""
entrenamiento_garch.py
SageMaker Processing Job — Entrenamiento ARMA-GARCH(1,1) Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Flujo:
  1. Cargar datos curated
  2. Split 80/20 cronológico
  3. Retornos log + ADF
  4. auto_arima sobre retornos → p_opt
  5. Ajustar GARCH sobre train → métricas in-sample
  6. Forecast 30 días desde fin de train → métricas OOS reales
  7. Reentrenar con serie completa → modelo de producción
  8. Guardar artefactos (pkl, metricas.json, metadata.json)
     con train_end = fin del TRAIN (no de la serie completa)
     para que inferencia sepa desde dónde arrancar el forecast de validación
"""

import subprocess, sys
for pkg in ["pmdarima", "pyarrow", "arch", "s3fs"]:
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
from arch import arch_model
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller, pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

INPUT_DIR  = Path("/opt/ml/processing/input/curated/embalse_guatape/")
OUTPUT_DIR = Path("/opt/ml/processing/output/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
LATEST_PREFIX   = "models/garch/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"models/garch/versions/{fecha_hoy}"

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR DATOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/8] Cargando datos curated...")

parquet_files = list(INPUT_DIR.rglob("*.parquet"))
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
print(f"  N       : {len(serie)} obs")
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
# 3. RETORNOS LOG + ADF
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/8] Retornos logarítmicos y ADF...")
retornos = np.log(train).diff().dropna()
adf = adfuller(retornos)
print(f"  ADF: stat={adf[0]:.4f}  p={adf[1]:.4f}"
      f"  → {'NO estacionaria' if adf[1] >= 0.05 else 'estacionaria'}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. AUTO_ARIMA → p_opt
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/8] auto_arima sobre retornos...")
N      = len(retornos)
umbral = 1.96 / np.sqrt(N)
pacf_v = pacf(retornos, nlags=40, method="ywm")
lags_p = [i for i in range(1, len(pacf_v)) if abs(pacf_v[i]) > umbral]
p_max  = min(len(lags_p), 7)

modelo_auto = auto_arima(
    retornos, start_p=0, start_q=0,
    max_p=p_max, max_q=3, d=0,
    seasonal=False, information_criterion="aic",
    trace=True, error_action="ignore",
    suppress_warnings=True, stepwise=True,
)
p_opt = modelo_auto.order[0]
print(f"\n  ARMA seleccionado: {modelo_auto.order}  →  AR lags={p_opt}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. AJUSTAR GARCH SOBRE TRAIN + MÉTRICAS IN-SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/8] Ajustando GARCH sobre train...")
ret_train_scaled = retornos * 100

garch_spec = arch_model(ret_train_scaled, mean="AR", lags=p_opt,
                        vol="Garch", p=1, q=1, dist="t")
res_train  = garch_spec.fit(update_freq=0, disp="off")
print(res_train.summary())

# In-sample: retornos ajustados → nivel
garch_fitted_ret = (ret_train_scaled - res_train.resid) / 100
prev_level       = train.shift(1)
fitted_level     = (prev_level * np.exp(garch_fitted_ret)).dropna()
idx_tr           = fitted_level.index.intersection(train.index)

mae_train  = float(mean_absolute_error(train.loc[idx_tr], fitted_level.loc[idx_tr]))
rmse_train = float(np.sqrt(mean_squared_error(train.loc[idx_tr], fitted_level.loc[idx_tr])))
mape_train = float((np.abs((train.loc[idx_tr] - fitted_level.loc[idx_tr])
                           / train.loc[idx_tr]) * 100).mean())

print(f"\n  In-sample → MAE: {mae_train:.4f}  RMSE: {rmse_train:.4f}  MAPE: {mape_train:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 6. MÉTRICAS OOS — forecast 30 días desde fin de train vs test real
#    Horizonte máximo evaluable = 30 días (GARCH es modelo de corto plazo)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[6/8] Métricas OOS (30 días desde fin de train)...")

H_EVAL    = min(30, len(test))
test_eval = test.iloc[:H_EVAL]

fc_oos    = res_train.forecast(horizon=H_EVAL)
mean_ret  = fc_oos.mean.iloc[-1].values / 100

prev_val  = train.iloc[-1]
niv_oos   = [prev_val]
for r in mean_ret:
    niv_oos.append(niv_oos[-1] * np.exp(r))

pred_oos = pd.Series(niv_oos[1:H_EVAL+1], index=test_eval.index)

mae_test  = float(mean_absolute_error(test_eval, pred_oos))
rmse_test = float(np.sqrt(mean_squared_error(test_eval, pred_oos)))
mape_test = float((np.abs((test_eval - pred_oos) / test_eval) * 100).mean())

print(f"  OOS ({H_EVAL}d) → MAE: {mae_test:.4f}  RMSE: {rmse_test:.4f}  MAPE: {mape_test:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 7. REENTRENAR CON SERIE COMPLETA → modelo de producción
# ══════════════════════════════════════════════════════════════════════════════
print("\n[7/8] Reentrenando con serie completa...")

ret_full_scaled = np.log(serie).diff().dropna() * 100
garch_spec_full = arch_model(ret_full_scaled, mean="AR", lags=p_opt,
                             vol="Garch", p=1, q=1, dist="t")
res_final = garch_spec_full.fit(update_freq=0, disp="off")

# ══════════════════════════════════════════════════════════════════════════════
# 8. GUARDAR ARTEFACTOS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[8/8] Guardando artefactos...")

pkl_path = OUTPUT_DIR / "modelo_garch.pkl"
with open(pkl_path, "wb") as f:
    pickle.dump(res_final, f)

metricas = {
    "mae_train"          : round(mae_train,  4),
    "rmse_train"         : round(rmse_train, 4),
    "mape_train"         : round(mape_train, 4),
    "mae_test"           : round(mae_test,   4),
    "rmse_test"          : round(rmse_test,  4),
    "mape_test"          : round(mape_test,  4),
    "oos_horizonte_dias" : H_EVAL,
    "nota_oos"           : f"OOS calculado sobre primeros {H_EVAL}d del test (forecast multi-step desde fin train)"
}
met_path = OUTPUT_DIR / "metricas.json"
met_path.write_text(json.dumps(metricas, indent=2))

metadata = {
    "embalse"             : "GUATAPE",
    "modelo"              : "ARMA-GARCH(1,1)",
    "ar_lags"             : p_opt,
    "garch_p"             : 1,
    "garch_q"             : 1,
    "dist"                : "t",
    "escala"              : "retornos_log_x100",
    "metrica"             : "VoluUtilDiarMasa",
    "unidad"              : "m3",
    "fecha_entrenamiento" : datetime.now(timezone.utc).isoformat(),
    "train_start"         : str(train.index.min().date()),
    "train_end"           : str(train.index.max().date()),   # ← fin del TRAIN, no de la serie
    "serie_end"           : str(serie.index.max().date()),   # ← fin de la serie completa
    "n_obs_train"         : int(len(train)),
    "n_obs_total"         : int(len(serie)),
}
meta_path = OUTPUT_DIR / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))

for artefacto in [pkl_path, met_path, meta_path]:
    for prefix in [LATEST_PREFIX, VERSIONS_PREFIX]:
        s3_upload(artefacto, f"{prefix}/{artefacto.name}")

print(f"\n✅ Entrenamiento GARCH completado.")
print(f"   train_end  : {metadata['train_end']}  (base métricas OOS)")
print(f"   serie_end  : {metadata['serie_end']}  (base forecast producción)")
print(f"   latest/    → s3://{BUCKET}/{LATEST_PREFIX}/")
print(f"   versions/  → s3://{BUCKET}/{VERSIONS_PREFIX}/")
