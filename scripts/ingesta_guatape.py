import sys
import subprocess

print("[inicio] Script arrancó correctamente")
print(f"[python] Versión: {sys.version}")

# Instalar pydataxm en tiempo de ejecución
print("[setup] Instalando pydataxm...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "pydataxm", "--quiet"
])
print("[setup] pydataxm instalada OK")

print("[import] Importando librerías...")
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import io
from datetime import datetime, timedelta
import pydataxm.pydataxm as pxm
print("[import] Todas las librerías OK")

# ─── Configuración ────────────────────────────────────────────────────────────
BUCKET         = "pi-2026"
RAW_PREFIX     = "data/raw/embalse_guatape/"
METADATA_KEY   = "data/raw/embalse_guatape/_metadata/ultima_descarga.txt"
FECHA_INICIO   = "2015-01-01"
CODIGO_EMBALSE = "PENOL"

VARIABLES = {
    "VoluUtilDiarMasa": "volumen_util_m3",
    "NivelEmbDiar":     "nivel_m",
    "AporteEmbDiar":    "aporte_m3s",
    "VertEmbDiar":      "vertimiento_m3s",
}

s3 = boto3.client("s3")
print("[aws] Cliente S3 creado OK")


# ─── Helpers ──────────────────────────────────────────────────────────────────
def leer_ultima_fecha():
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=METADATA_KEY)
        fecha = obj["Body"].read().decode("utf-8").strip()
        print(f"[metadata] Última fecha descargada: {fecha}")
        return fecha
    except Exception:
        print("[metadata] No existe metadata — modo histórico.")
        return None


def guardar_ultima_fecha(fecha):
    s3.put_object(
        Bucket=BUCKET,
        Key=METADATA_KEY,
        Body=fecha.encode("utf-8")
    )
    print(f"[metadata] Actualizada última fecha: {fecha}")


def subir_parquet(df, variable_nombre):
    if df.empty:
        print(f"[{variable_nombre}] DataFrame vacío, se omite.")
        return

    df["Date"] = pd.to_datetime(df["Date"])
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month

    for (year, month), grupo in df.groupby(["year", "month"]):
        grupo_limpio = grupo.drop(columns=["year", "month"])
        buffer = io.BytesIO()
        tabla = pa.Table.from_pandas(grupo_limpio, preserve_index=False)
        pq.write_table(tabla, buffer)
        buffer.seek(0)

        key = f"{RAW_PREFIX}{variable_nombre}/year={year}/month={month:02d}/data.parquet"
        s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
        print(f"  → s3://{BUCKET}/{key} ({len(grupo_limpio)} registros)")


# ─── Lógica principal ─────────────────────────────────────────────────────────
def main():
    hoy    = datetime.today().strftime("%Y-%m-%d")
    ultima = leer_ultima_fecha()

    if ultima is None:
        fecha_desde = FECHA_INICIO
        print(f"\n[modo] HISTÓRICO: {fecha_desde} → {hoy}\n")
    else:
        fecha_desde = (
            datetime.strptime(ultima, "%Y-%m-%d") + timedelta(days=1)
        ).strftime("%Y-%m-%d")

        if fecha_desde > hoy:
            print("[info] Los datos ya están al día.")
            return

        print(f"\n[modo] INCREMENTAL: {fecha_desde} → {hoy}\n")

    print("[xm] Conectando con API de XM...")
    obj = pxm.ReadDB()
    print("[xm] Conexión OK")
    fechas_maximas = []

    for codigo_variable, nombre in VARIABLES.items():
        print(f"[descarga] {nombre} ({codigo_variable})...")
        try:
            df = obj.request_data(
                codigo_variable, "Embalse",
                fecha_desde, hoy,
                filtros=[CODIGO_EMBALSE]
            )

            if df is None or df.empty:
                print(f"  [!] Sin datos para {nombre}.")
                continue

            print(f"  {len(df)} registros descargados.")
            subir_parquet(df, nombre)

            df["Date"] = pd.to_datetime(df["Date"])
            fechas_maximas.append(df["Date"].max().strftime("%Y-%m-%d"))

        except Exception as e:
            print(f"  [ERROR] {nombre}: {e}")

    if fechas_maximas:
        guardar_ultima_fecha(max(fechas_maximas))

    print("\n[ok] Ingesta finalizada.")


print("[main] Llamando main()...")
main()
print("[main] main() terminó.")