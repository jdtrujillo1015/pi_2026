"""
inferencia_garch.py
SageMaker Processing Job — Inferencia ARMA-GARCH(1,1) Guatapé
Imagen: sagemaker-scikit-learn:1.2-1-cpu-py3 (Python 3.9)

Inputs  (ProcessingInput):
    /opt/ml/processing/input/model/    ← models/garch/latest/
    /opt/ml/processing/input/curated/  ← data/curated/embalse_guatape/volumen_curated.parquet
        Local path consola: /opt/ml/processing/input/curated
        S3 location consola: s3://pi-2026/data/curated/embalse_guatape/volumen_curated.parquet

Outputs (ProcessingOutput):
    /opt/ml/processing/output/preds/   → predictions/garch/latest/
                                         predictions/garch/versions/<fecha>/
"""

import subprocess, sys
for pkg in ["pyarrow", "arch", "s3fs"]:
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
INPUT_MODEL   = Path("/opt/ml/processing/input/model")
INPUT_CURATED = Path("/opt/ml/processing/input/curated")   # raíz limpia, sin subcarpetas
OUTPUT_DIR    = Path("/opt/ml/processing/output/preds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BUCKET          = "pi-2026"
S3_CURATED      = f"s3://{BUCKET}/data/curated/embalse_guatape/volumen_curated.parquet"
LATEST_PREFIX   = "predictions/garch/latest"
fecha_hoy       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
VERSIONS_PREFIX = f"predictions/garch/versions/{fecha_hoy}"
HORIZONTES      = [7, 15, 30]

s3 = boto3.client("s3")

def s3_upload(local_path: Path, s3_key: str):
    s3.upload_file(str(local_path), BUCKET, s3_key)
    print(f"  ✅ s3://{BUCKET}/{s3_key}")

# ══════════════════════════════════════════════════════════════════════════════
# 1. CARGAR MODELO Y METADATA
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/4] Cargando modelo y metadata...")

with open(INPUT_MODEL / "modelo_garch.pkl", "rb") as f:
    model = pickle.load(f)

metadata = json.loads((INPUT_MODEL / "metadata.json").read_text())
metricas = json.loads((INPUT_MODEL / "metricas.json").read_text())

print(f"  Modelo      : {metadata['modelo']} AR({metadata['ar_lags']})")
print(f"  Entrenado   : {metadata['fecha_entrenamiento']}")
print(f"  Datos hasta : {metadata['train_end']}")
print(f"  MAE test    : {metricas['mae_test']}")
print(f"  MAPE test   : {metricas['mape_test']:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# 2. CARGAR ÚLTIMO VALOR OBSERVADO
#    Estrategia: buscar el parquet en cualquier nivel bajo INPUT_CURATED.
#    Si no aparece, leer directo desde S3 (fallback garantizado).
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n[2/4] Cargando último dato observado...")

# Diagnóstico completo del directorio montado
print(f"  Contenido de {INPUT_CURATED}:")
todos = list(INPUT_CURATED.rglob("*"))
if todos:
    for p in sorted(todos):
        tag = "[DIR]" if p.is_dir() else "     "
        print(f"    {tag} {p}")
else:
    print("    (vacío)")

# Buscar parquet en cualquier nivel de profundidad
parquet_files = list(INPUT_CURATED.rglob("*.parquet"))
print(f"\n  Parquets encontrados en input: {len(parquet_files)}")

if parquet_files:
    print(f"  Leyendo desde: {parquet_files[0]}")
    df_c = pd.read_parquet(parquet_files[0])
else:
    # Fallback directo a S3 — siempre funciona con LabRole
    print(f"  ⚠ Input vacío — leyendo desde S3: {S3_CURATED}")
    import s3fs
    fs = s3fs.S3FileSystem()
    with fs.open(S3_CURATED) as f_s3:
        df_c = pd.read_parquet(f_s3)

df_c["fecha"]  = pd.to_datetime(df_c["fecha"])
df_c           = df_c.set_index("fecha").sort_index()
serie_completa = df_c["volumen_m3"].dropna()

ultimo_val   = float(serie_completa.iloc[-1])
ultima_fecha = serie_completa.index[-1]
fecha_base   = ultima_fecha + timedelta(days=1)

print(f"  Último valor : {ultimo_val:,.4f} m³  ({ultima_fecha.date()})")
print(f"  Desde        : {fecha_base.date()}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. GENERAR FORECASTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/4] Generando forecasts GARCH...")

archivos_generados = []

for h in HORIZONTES:
    fc_res   = model.forecast(horizon=h)
    mean_ret = fc_res.mean.iloc[-1].values / 100       # desescalar ÷ 100
    var_ret  = fc_res.variance.iloc[-1].values / 1e4   # desescalar ÷ 100²
    std_ret  = np.sqrt(var_ret)

    ci_lo = mean_ret - 1.96 * std_ret
    ci_hi = mean_ret + 1.96 * std_ret

    # Reconvertir retornos → niveles: P_t = P_{t-1} × exp(r_t)
    niv, niv_lo, niv_hi = [ultimo_val], [ultimo_val], [ultimo_val]
    for r, rl, ru in zip(mean_ret, ci_lo, ci_hi):
        niv.append(   niv[-1]    * np.exp(r))
        niv_lo.append(niv_lo[-1] * np.exp(rl))
        niv_hi.append(niv_hi[-1] * np.exp(ru))

    fechas  = pd.date_range(start=fecha_base, periods=h, freq="D")
    df_pred = pd.DataFrame({
        "fecha"               : fechas.strftime("%Y-%m-%d"),
        "volumen_predicho_m3" : np.round(niv[1:],    4),
        "varianza_condicional": np.round(var_ret,     8),
        "ic_lower_95"         : np.round(niv_lo[1:], 4),
        "ic_upper_95"         : np.round(niv_hi[1:], 4),
    })

    nombre     = f"forecast_{h}d.parquet"
    local_path = OUTPUT_DIR / nombre
    df_pred.to_parquet(local_path, index=False)
    archivos_generados.append((nombre, local_path))

    print(f"\n  Horizonte {h:2d} días → {fechas[0].date()} … {fechas[-1].date()}")
    print(f"    Media : {np.mean(niv[1:]):>12,.4f} m³")
    print(f"    Min   : {np.min(niv[1:]):>12,.4f} m³")
    print(f"    Max   : {np.max(niv[1:]):>12,.4f} m³")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SUBIR A S3 + RESUMEN
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/4] Subiendo a S3...")
for nombre, local_path in archivos_generados:
    for prefix in [LATEST_PREFIX, VERSIONS_PREFIX]:
        s3_upload(local_path, f"{prefix}/{nombre}")

print(f"\n  latest/   → s3://{BUCKET}/{LATEST_PREFIX}/")
print(f"  versions/ → s3://{BUCKET}/{VERSIONS_PREFIX}/")
for h in HORIZONTES:
    print(f"  forecast_{h}d.parquet → fecha | volumen_predicho_m3 | varianza_condicional | ic_lower_95 | ic_upper_95")
print("\n✅ Inferencia GARCH completada.")
