import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from config import RAW_DIR, CURATED_DIR

# Leer
df = pd.read_csv(os.path.join(RAW_DIR, "volumen_m3.csv"))

# Quedarse solo con fecha y valor
df = df[["Date", "Value"]].copy()
df.columns = ["fecha", "volumen_m3"]

# Fecha como tipo datetime
df["fecha"] = pd.to_datetime(df["fecha"])
df = df.sort_values("fecha").reset_index(drop=True)

# Verificar nulos
nulos = df["volumen_m3"].isnull().sum()
print(f"Valores nulos: {nulos}")
if nulos > 0:
    df["volumen_m3"] = df["volumen_m3"].interpolate(method="linear")
    print("Nulos interpolados.")

# Verificar duplicados
duplicados = df["fecha"].duplicated().sum()
print(f"Fechas duplicadas: {duplicados}")
df = df.drop_duplicates(subset="fecha")

# Guardar
ruta = os.path.join(CURATED_DIR, "volumen_m3.csv")
df.to_csv(ruta, index=False, encoding="utf-8")
print(f"Curated guardado: {ruta} ({len(df)} registros)")
print(df.head())