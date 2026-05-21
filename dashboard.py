import os
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from config import CURATED_DIR, PREDICTIONS_DIR, ARTIFACTS_DIR

st.set_page_config(page_title="Embalse El Peñol", layout="wide")
st.title("Embalse El Peñol — Volumen útil diario")

# Cargar datos
historico = pd.read_csv(os.path.join(CURATED_DIR, "volumen_m3.csv"))
historico["fecha"] = pd.to_datetime(historico["fecha"])

archivos = sorted(os.listdir(PREDICTIONS_DIR))
if not archivos:
    st.error("No hay predicciones disponibles.")
    st.stop()

ultimo = archivos[-1]
prediccion = pd.read_csv(os.path.join(PREDICTIONS_DIR, ultimo))
prediccion["fecha"] = pd.to_datetime(prediccion["fecha"])

with open(os.path.join(ARTIFACTS_DIR, "metrics.json")) as f:
    metrics = json.load(f)

# Sección 1: estado actual y predicción
st.subheader("Estado actual y predicción a 7 días")
col1, col2, col3 = st.columns(3)
col1.metric("Último volumen registrado", f"{historico['volumen_m3'].iloc[-1]/1e6:.1f} Mm³")
col2.metric("Predicción día 1", f"{prediccion['volumen_m3_predicho'].iloc[0]/1e6:.1f} Mm³")
col3.metric("Predicción día 7", f"{prediccion['volumen_m3_predicho'].iloc[-1]/1e6:.1f} Mm³")

# Sección 2: gráfico
fig, ax = plt.subplots(figsize=(14, 5))
historico_reciente = historico.tail(180)
ax.plot(historico_reciente["fecha"], historico_reciente["volumen_m3"] / 1e6,
        label="Histórico", color="#1D9E75", linewidth=1.5)
ax.plot(prediccion["fecha"], prediccion["volumen_m3_predicho"] / 1e6,
        label="Predicción 7 días", color="#534AB7", linewidth=1.5, linestyle="--")
ax.axvline(x=historico["fecha"].iloc[-1], color="#D85A30", linewidth=1, linestyle=":")
ax.set_ylabel("Volumen (Mm³)")
ax.set_xlabel("Fecha")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)
st.caption(f"Predicción generada: {ultimo.replace('prediccion_','').replace('.csv','')}")

# Sección 3: métricas del modelo
st.subheader("Rendimiento del modelo (test set)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MAE",  f"{metrics['mae']/1e6:.1f} Mm³")
col2.metric("RMSE", f"{metrics['rmse']/1e6:.1f} Mm³")
col3.metric("MAPE", f"{metrics['mape']:.2f}%")
col4.metric("R²",   f"{metrics['r2']:.4f}")