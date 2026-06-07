import sys
import subprocess

print("[inicio] Script de procesamiento arrancó")
print(f"[python] Versión: {sys.version}")

print("[setup] Instalando dependencias...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "pyarrow", "s3fs", "--quiet"])
print("[setup] Dependencias OK")

import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import s3fs
import io

# ─── Configuración ────────────────────────────────────────────────────────────
BUCKET      = "pi-2026"
RAW_PATH    = "pi-2026/data/raw/embalse_guatape/volumen_util_m3/"
CURATED_KEY = "data/curated/embalse_guatape/volumen_curated.parquet"

s3 = boto3.client("s3")
fs = s3fs.S3FileSystem()

# ─── 1. Leer datos raw ────────────────────────────────────────────────────────
print("\n[paso 1] Leyendo datos desde S3 raw...")
dataset = pq.ParquetDataset(RAW_PATH, filesystem=fs)
df = dataset.read().to_pandas()
print(f"  Registros leídos: {len(df)}")

# ─── 2. Seleccionar y renombrar columnas relevantes ───────────────────────────
print("\n[paso 2] Limpieza básica...")
df = df[["Date", "Value"]].copy()
df.columns = ["fecha", "volumen_m3"]
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)
print(f"  Shape: {df.shape}")
print(f"  Nulos: {df.isnull().sum().to_dict()}")

# ─── 3. Completar rango temporal e imputar día faltante ──────────────────────
print("\n[paso 3] Imputando día faltante...")
rango_completo = pd.date_range(
    start=df["fecha"].min(),
    end=df["fecha"].max(),
    freq="D"
)
df = df.set_index("fecha").reindex(rango_completo).rename_axis("fecha").reset_index()
nulos = df["volumen_m3"].isnull().sum()
df["volumen_m3"] = df["volumen_m3"].interpolate(method="linear")
print(f"  Días imputados: {nulos}")
print(f"  Nulos restantes: {df['volumen_m3'].isnull().sum()}")
print(f"  Rango: {df['fecha'].min().date()} → {df['fecha'].max().date()}")
print(f"  Total registros: {len(df)}")

# ─── 4. Verificación final ────────────────────────────────────────────────────
print("\n[paso 4] Verificación final...")
print(f"  Volumen mín : {df['volumen_m3'].min():.0f} m³")
print(f"  Volumen máx : {df['volumen_m3'].max():.0f} m³")
print(f"  Volumen medio: {df['volumen_m3'].mean():.0f} m³")
assert df["volumen_m3"].isnull().sum() == 0, "Aún hay nulos en volumen_m3"
assert df["fecha"].isnull().sum() == 0, "Aún hay nulos en fecha"
print("  Validaciones OK ✓")

# ─── 5. Guardar en S3 curated ─────────────────────────────────────────────────
print("\n[paso 5] Guardando en S3 curated...")
buffer = io.BytesIO()
tabla = pa.Table.from_pandas(df, preserve_index=False)
pq.write_table(tabla, buffer)
buffer.seek(0)
s3.put_object(Bucket=BUCKET, Key=CURATED_KEY, Body=buffer.getvalue())
print(f"  → s3://{BUCKET}/{CURATED_KEY}")
print(f"  Registros guardados: {len(df)}")
print(f"  Columnas: {df.columns.tolist()}")

print("\n[ok] Procesamiento finalizado.")