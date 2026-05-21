import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from datetime import datetime

import pydataxm.pydataxm as pxm
from config import RAW_DIR

FECHA_INICIO   = "2015-01-01"
FECHA_FIN      = datetime.today().strftime("%Y-%m-%d")
CODIGO_EMBALSE = "PENOL"

obj = pxm.ReadDB()

print("Descargando volumen útil diario (m3)...")
volumen = obj.request_data(
    "VoluUtilDiarMasa", "Embalse",
    FECHA_INICIO, FECHA_FIN,
    filtros=[CODIGO_EMBALSE]
)

ruta = os.path.join(RAW_DIR, "volumen_m3.csv")
volumen.to_csv(ruta, index=False, encoding="utf-8")
print(f"Guardado: {ruta} ({len(volumen)} registros)")