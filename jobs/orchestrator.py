import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from config import REENTRENAR

import schedule
import time
import subprocess
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))

def run_script(nombre):
    ruta = os.path.join(BASE, nombre)
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Ejecutando {nombre}...")
    result = subprocess.run([sys.executable, ruta], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"ERROR en {nombre}:")
        print(result.stderr)

def pipeline_diario():
    print(f"\n{'='*50}")
    print(f"Pipeline iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*50}")
    run_script("descarga.py")
    run_script("curate.py")
    if REENTRENAR:
        run_script("featurize.py")
        run_script("train.py")
    run_script("infer.py")
    print(f"\nPipeline completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Correr todos los días a las 6:00 AM
schedule.every().day.at("06:00").do(pipeline_diario)

print("Orquestador activo. Esperando las 06:00...")
print("Presiona Ctrl+C para detener.")

# Ejecutar una vez al arrancar para verificar
pipeline_diario()

while True:
    schedule.run_pending()
    time.sleep(60)