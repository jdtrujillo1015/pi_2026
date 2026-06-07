# Estimación del volumen de agua del embalse de Guatapé
### Proyecto Integrador II — Maestría en Ciencia de Datos y Analítica
**Universidad EAFIT · Escuela de Ciencias Aplicadas e Ingeniería · 2026**

> Franco Garcés · Isabel Jurado · Dilan Monsalve · Alejandro Silva · Juan David Trujillo

---

## Dashboard

🔗 [Ver dashboard en vivo](http://pi-2026-dashboard-387623072987.s3-website-us-east-1.amazonaws.com/index.html)

Actualización automática diaria a las 12pm (hora Colombia). Muestra serie histórica, pronósticos de los 4 modelos con intervalos de confianza al 95% y métricas comparativas.

---

## Descripción

Sistema predictivo para estimar el volumen útil diario del embalse de El Peñol-Guatapé usando datos históricos de la API pública de XM. Integra cuatro modelos de forecasting desplegados en un pipeline MLOps automatizado en AWS.

### Modelos implementados

| Modelo | MAPE test | MAE test | RMSE test |
|--------|-----------|----------|-----------|
| Holt-Winters (ETS) | 0.57% | 2.55 Mm³ | 4.06 Mm³ |
| LSTM (TensorFlow/Keras) | 0.95% | 5.92 Mm³ | 7.51 Mm³ |
| ARMA-GARCH(1,1) | 6.18% | 28.21 Mm³ | 34.13 Mm³ |
| ARIMA | 18.12% | 83.76 Mm³ | 95.27 Mm³ |

Split cronológico 80/20. Evaluación out-of-sample sobre el 20% final de la serie (~4200 observaciones diarias desde 2015).

---

## Arquitectura del pipeline

```
EventBridge (12pm Colombia)
        │
        ▼
SageMaker Pipeline: pipeline-guatape (11 steps)
        │
        ├── 1. ingesta-guatape
        │       └── API XM → S3 raw (incremental)
        │
        ├── 2. procesamiento-guatape
        │       └── raw → curated (limpieza + imputación)
        │
        ├── 3. entrenamiento-arima ──→ 4. inferencia-arima
        ├── 5. entrenamiento-garch ──→ 6. inferencia-garch    (paralelo)
        ├── 7. entrenamiento-hw    ──→ 8. inferencia-hw
        └── 9. entrenamiento-lstm  ──→ 10. inferencia-lstm
                                              │
                                        11. dashboard-guatape
                                              └── HTML → S3 público

SNS → email de notificación al terminar
```

### Infraestructura AWS

| Servicio | Uso |
|----------|-----|
| S3 `pi-2026` | Datos, modelos, predicciones, scripts |
| S3 `pi-2026-dashboard-*` | Dashboard HTML público |
| SageMaker Processing Jobs | Ejecución de cada step |
| SageMaker Pipelines | Orquestación de los 11 steps |
| Amazon EventBridge | Disparo automático diario |
| Amazon SNS | Notificaciones por email |

---

## Estructura del repositorio

```
pi_2026/
├── scripts/
│   ├── ingesta_guatape.py
│   ├── procesamiento_guatape.py
│   ├── entrenamiento_arima.py
│   ├── inferencia_arima.py
│   ├── entrenamiento_garch.py
│   ├── inferencia_garch.py
│   ├── entrenamiento_hw.py
│   ├── inferencia_hw.py
│   ├── entrenamiento_lstm.py
│   ├── inferencia_lstm.py
│   └── dashboard.py
├── notebooks/
│   ├── EDA.ipynb
│   └── pipeline_orquestador.ipynb
├── docs/
│   └── entrega_final.pdf
└── README.md
```

---

## Reproducción del pipeline

**Requisitos:** cuenta AWS con SageMaker, S3, EventBridge y SNS. Rol IAM con permisos sobre estos servicios.

**1. Subir scripts a S3**
```bash
aws s3 sync scripts/ s3://pi-2026/scripts/
```

**2. Abrir la instancia de notebook en SageMaker**

Kernel: `conda_python3`

**3. Ejecutar `pipeline_orquestador.ipynb`**

Las celdas de configuración, definición de steps y `pipeline.upsert()` registran el pipeline en SageMaker. `pipeline.start()` lanza una ejecución manual.

**4. Verificar en EventBridge**

La regla `pipeline-guatape-diario` dispara automáticamente el pipeline todos los días a las 17:00 UTC (12pm Colombia).

---

## Fuente de datos

- **API XM** — `pydataxm`, colección `VoluUtilDiarMasa`, métrica `Embalse`, código `PENOL`
- Acceso público y gratuito
- Datos desde 2015-01-01, frecuencia diaria
