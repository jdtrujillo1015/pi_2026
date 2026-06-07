"""
dashboard.py
SageMaker Processing Job — Genera dashboard HTML con predicciones del embalse Guatapé
y lo sube a S3 con acceso público.

Imagen: sagemaker-scikit-learn:1.4-2-cpu-py3
Sin inputs/outputs de Processing — lee y escribe directamente en S3.
"""

import subprocess, sys
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyarrow', '-q'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

import boto3
import json
import io
from datetime import datetime, timezone
import pyarrow.parquet as pq

# ── Configuración ─────────────────────────────────────────────────────────────
BUCKET           = 'pi-2026'
DASHBOARD_BUCKET = 'pi-2026-dashboard-387623072987'
DASHBOARD_KEY    = 'index.html'

s3 = boto3.client('s3')
print('[INFO] Leyendo datos desde S3...')

def leer_parquet(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pq.read_table(io.BytesIO(obj['Body'].read())).to_pandas()

# Histórico completo (para filtros de tiempo)
df_hist_full = leer_parquet('data/curated/embalse_guatape/volumen_curated.parquet')
df_hist_full['fecha'] = df_hist_full['fecha'].astype(str)

# ARIMA
df_arima_7  = leer_parquet('predictions/arima/latest/forecast_7d.parquet')
df_arima_15 = leer_parquet('predictions/arima/latest/forecast_15d.parquet')
df_arima_30 = leer_parquet('predictions/arima/latest/forecast_30d.parquet')
for df in [df_arima_7, df_arima_15, df_arima_30]:
    df['fecha'] = df['fecha'].astype(str)

# GARCH
df_garch_7  = leer_parquet('predictions/garch/latest/forecast_7d.parquet')
df_garch_15 = leer_parquet('predictions/garch/latest/forecast_15d.parquet')
df_garch_30 = leer_parquet('predictions/garch/latest/forecast_30d.parquet')
for df in [df_garch_7, df_garch_15, df_garch_30]:
    df['fecha'] = df['fecha'].astype(str)

# Holt-Winters
df_hw_7  = leer_parquet('predictions/hw/latest/forecast_7d.parquet')
df_hw_15 = leer_parquet('predictions/hw/latest/forecast_15d.parquet')
df_hw_30 = leer_parquet('predictions/hw/latest/forecast_30d.parquet')
for df in [df_hw_7, df_hw_15, df_hw_30]:
    df['fecha'] = df['fecha'].astype(str)

# LSTM
df_lstm_7  = leer_parquet('predictions/lstm/latest/forecast_7d.parquet')
df_lstm_15 = leer_parquet('predictions/lstm/latest/forecast_15d.parquet')
df_lstm_30 = leer_parquet('predictions/lstm/latest/forecast_30d.parquet')
for df in [df_lstm_7, df_lstm_15, df_lstm_30]:
    df['fecha'] = df['fecha'].astype(str)

# Métricas
metricas_arima = json.loads(s3.get_object(Bucket=BUCKET, Key='models/arima/latest/metricas.json')['Body'].read())
metricas_garch = json.loads(s3.get_object(Bucket=BUCKET, Key='models/garch/latest/metricas.json')['Body'].read())
metricas_hw    = json.loads(s3.get_object(Bucket=BUCKET, Key='models/hw/latest/metricas.json')['Body'].read())
metricas_lstm  = json.loads(s3.get_object(Bucket=BUCKET, Key='models/lstm/latest/metricas.json')['Body'].read())

print('[INFO] Datos leídos OK')
print(f'  Histórico  : {len(df_hist_full)} días')
print(f'  Modelos    : ARIMA / GARCH / HW / LSTM — 7/15/30 días')

def df_to_js(df):
    return json.dumps(df.to_dict(orient='list'))

fecha_actualizacion = datetime.now(timezone.utc).strftime('%d/%m/%Y %H:%M UTC')

html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Embalse Guatapé — Monitor de Volumen</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:      #0b0f1a;
    --surface: #111827;
    --border:  #1e2d40;
    --text:    #e2e8f0;
    --muted:   #64748b;
    --accent:  #38bdf8;
    --lstm:    #34d399;
    --arima:   #f59e0b;
    --garch:   #a78bfa;
    --hw:      #fb7185;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; min-height: 100vh; }}

  header {{
    border-bottom: 1px solid var(--border);
    padding: 1.5rem 3rem;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
    background: linear-gradient(135deg, #0b0f1a 0%, #0f1e2e 100%);
  }}
  header h1 {{ font-family: 'DM Serif Display', serif; font-size: 1.8rem; font-weight: 400; line-height: 1.1; }}
  header h1 span {{ color: var(--accent); font-style: italic; }}
  .update-badge {{ font-size: 0.7rem; color: var(--muted); text-align: right; line-height: 1.8; }}
  .update-badge strong {{ color: var(--accent); font-weight: 500; }}

  main {{ padding: 2rem 3rem; max-width: 1400px; margin: 0 auto; }}

  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(6, 1fr);
    gap: 0.75rem;
    margin-bottom: 2rem;
  }}
  .kpi {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.1rem 1.2rem;
    position: relative;
    overflow: hidden;
  }}
  .kpi::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; }}
  .kpi.accent::before {{ background: var(--accent); }}
  .kpi.lstm::before   {{ background: var(--lstm); }}
  .kpi.arima::before  {{ background: var(--arima); }}
  .kpi.garch::before  {{ background: var(--garch); }}
  .kpi.hw::before     {{ background: var(--hw); }}
  .kpi.muted::before  {{ background: var(--muted); }}
  .kpi-label {{ font-size: 0.6rem; color: var(--muted); letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem; }}
  .kpi-value {{ font-family: 'DM Serif Display', serif; font-size: 1.5rem; font-weight: 400; line-height: 1; }}
  .kpi-sub   {{ font-size: 0.6rem; color: var(--muted); margin-top: 0.3rem; }}

  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.5rem;
  }}
  .chart-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    flex-wrap: wrap;
    gap: 0.5rem;
  }}
  .chart-title {{ font-family: 'DM Serif Display', serif; font-size: 1.1rem; font-weight: 400; }}
  .legend {{ display: flex; gap: 1rem; font-size: 0.65rem; color: var(--muted); flex-wrap: wrap; }}
  .legend-item {{ display: flex; align-items: center; gap: 0.4rem; }}
  .legend-dot {{ width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }}

  .controls {{ display: flex; gap: 0.4rem; flex-wrap: wrap; }}
  .tab {{
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    padding: 0.35rem 0.8rem;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.15s;
    letter-spacing: 0.05em;
  }}
  .tab:hover  {{ border-color: var(--accent); color: var(--text); }}
  .tab.active {{ background: var(--accent); border-color: var(--accent); color: #0b0f1a; font-weight: 500; }}

  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .metrics-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.4rem;
  }}
  .metrics-card h3 {{
    font-family: 'DM Serif Display', serif;
    font-size: 0.95rem;
    font-weight: 400;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }}
  .model-tag {{
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    padding: 0.15rem 0.4rem;
    border-radius: 4px;
    letter-spacing: 0.1em;
  }}
  .tag-lstm  {{ background: rgba(52,211,153,0.15);  color: var(--lstm); }}
  .tag-arima {{ background: rgba(245,158,11,0.15);  color: var(--arima); }}
  .tag-garch {{ background: rgba(167,139,250,0.15); color: var(--garch); }}
  .tag-hw    {{ background: rgba(251,113,133,0.15); color: var(--hw); }}
  .metric-row {{
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.72rem;
  }}
  .metric-row:last-child {{ border-bottom: none; }}
  .metric-name  {{ color: var(--muted); }}
  .metric-value {{ font-weight: 500; }}

  footer {{ text-align: center; padding: 1.5rem; font-size: 0.6rem; color: var(--muted); border-top: 1px solid var(--border); letter-spacing: 0.05em; }}
  canvas {{ max-height: 340px; }}

  @media (max-width: 900px) {{
    header {{ flex-direction: column; align-items: flex-start; gap: 0.8rem; padding: 1rem 1.2rem; }}
    main {{ padding: 1rem; }}
    .kpi-grid {{ grid-template-columns: repeat(3, 1fr); }}
    .metrics-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .chart-header {{ flex-direction: column; align-items: flex-start; }}
  }}
  @media (max-width: 500px) {{
    .kpi-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .metrics-grid {{ grid-template-columns: 1fr; }}
    canvas {{ max-height: 220px; }}
  }}
</style>
</head>
<body>

<header>
  <div>
    <h1>Embalse <span>Guatapé</span><br>Monitor de Volumen</h1>
  </div>
  <div class="update-badge">
    Actualización automática diaria<br>
    Última actualización: <strong>{fecha_actualizacion}</strong>
  </div>
</header>

<main>

  <!-- KPIs -->
  <div class="kpi-grid" id="kpi-grid"></div>

  <!-- Serie histórica -->
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Serie histórica y pronóstico</span>
      <div style="display:flex; gap:1rem; align-items:center; flex-wrap:wrap;">
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:var(--accent)"></div>Histórico</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--lstm)"></div>LSTM</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--arima)"></div>ARIMA</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--garch)"></div>GARCH</div>
          <div class="legend-item"><div class="legend-dot" style="background:var(--hw)"></div>HW</div>
        </div>
        <div class="controls">
          <button class="tab" onclick="setRange(7,this)">1S</button>
          <button class="tab" onclick="setRange(30,this)">1M</button>
          <button class="tab active" onclick="setRange(365,this)">1A</button>
          <button class="tab" onclick="setRange(1825,this)">5A</button>
        </div>
      </div>
    </div>
    <canvas id="chartHistorico"></canvas>
  </div>

  <!-- IC por modelo -->
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Pronóstico con intervalo de confianza 95%</span>
      <div style="display:flex; gap:0.5rem; flex-wrap:wrap;">
        <div class="controls" style="margin-right:0.5rem;">
          <button class="tab-model tab active" onclick="switchModel('arima',this)">ARIMA</button>
          <button class="tab-model tab" onclick="switchModel('garch',this)">GARCH</button>
          <button class="tab-model tab" onclick="switchModel('hw',this)">HW</button>
          <button class="tab-model tab" onclick="switchModel('lstm',this)">LSTM</button>
        </div>
        <div class="controls">
          <button class="tab-ic tab active" onclick="switchIC(7,this)">7 días</button>
          <button class="tab-ic tab" onclick="switchIC(15,this)">15 días</button>
          <button class="tab-ic tab" onclick="switchIC(30,this)">30 días</button>
        </div>
      </div>
    </div>
    <canvas id="chartIC"></canvas>
  </div>

  <!-- Comparativo -->
  <div class="chart-card">
    <div class="chart-header">
      <span class="chart-title">Comparativo de modelos</span>
      <div class="controls">
        <button class="tab-comp tab active" onclick="switchComp(7,this)">7 días</button>
        <button class="tab-comp tab" onclick="switchComp(15,this)">15 días</button>
        <button class="tab-comp tab" onclick="switchComp(30,this)">30 días</button>
      </div>
    </div>
    <canvas id="chartComparativo"></canvas>
  </div>

  <!-- Métricas -->
  <div class="metrics-grid" id="metrics-grid"></div>

</main>

<footer>MAESTRÍA EN CIENCIA DE DATOS Y ANALÍTICA · UNIVERSIDAD EAFIT · PROYECTO INTEGRADOR II 2026</footer>

<script>
// ── Datos ────────────────────────────────────────────────────────────────────
const histFull = {df_to_js(df_hist_full)};
const arima7   = {df_to_js(df_arima_7)};
const arima15  = {df_to_js(df_arima_15)};
const arima30  = {df_to_js(df_arima_30)};
const garch7   = {df_to_js(df_garch_7)};
const garch15  = {df_to_js(df_garch_15)};
const garch30  = {df_to_js(df_garch_30)};
const hw7      = {df_to_js(df_hw_7)};
const hw15     = {df_to_js(df_hw_15)};
const hw30     = {df_to_js(df_hw_30)};
const lstm7    = {df_to_js(df_lstm_7)};
const lstm15   = {df_to_js(df_lstm_15)};
const lstm30   = {df_to_js(df_lstm_30)};
const metARIMA = {json.dumps(metricas_arima)};
const metGARCH = {json.dumps(metricas_garch)};
const metHW    = {json.dumps(metricas_hw)};
const metLSTM  = {json.dumps(metricas_lstm)};

// ── KPIs ─────────────────────────────────────────────────────────────────────
function fmt(v) {{ return v != null ? (v/1e6).toFixed(1) + ' Mm³' : 'N/D'; }}

const volActual = histFull.volumen_m3[histFull.volumen_m3.length - 1];
const fechaUlt  = histFull.fecha[histFull.fecha.length - 1];

const kpis = [
  {{ label: 'Volumen actual',      value: fmt(volActual),                                           sub: fechaUlt,           cls: 'accent' }},
  {{ label: 'Pronóstico LSTM +7d', value: fmt(lstm7.volumen_predicho_m3[6]),                        sub: 'Red neuronal',     cls: 'lstm'   }},
  {{ label: 'Pronóstico ARIMA +7d',value: fmt(arima7.volumen_predicho_m3[6]),                       sub: 'ARIMA',            cls: 'arima'  }},
  {{ label: 'Pronóstico GARCH +7d',value: fmt(garch7.volumen_predicho_m3[6]),                       sub: 'ARMA-GARCH(1,1)',  cls: 'garch'  }},
  {{ label: 'Pronóstico HW +7d',   value: fmt(hw7.volumen_predicho_m3[6]),                          sub: 'Holt-Winters',     cls: 'hw'     }},
  {{ label: 'Días de histórico',   value: histFull.fecha.length,                                    sub: 'Serie completa',   cls: 'muted'  }},
];
const kpiGrid = document.getElementById('kpi-grid');
kpis.forEach(k => {{
  kpiGrid.innerHTML += `<div class="kpi ${{k.cls}}">
    <div class="kpi-label">${{k.label}}</div>
    <div class="kpi-value">${{k.value}}</div>
    <div class="kpi-sub">${{k.sub}}</div>
  </div>`;
}});

// ── Chart.js defaults ─────────────────────────────────────────────────────────
Chart.defaults.color = '#64748b';
Chart.defaults.borderColor = '#1e2d40';
Chart.defaults.font.family = 'DM Mono, monospace';
Chart.defaults.font.size = 11;

const tooltipDefaults = {{
  backgroundColor: '#111827',
  borderColor: '#1e2d40',
  borderWidth: 1,
  callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y?.toFixed(2)}} Mm³` }}
}};

// ── Gráfica histórica con filtro de tiempo ────────────────────────────────────
let histChart = null;
let currentRange = 365;

function buildHistChart(days) {{
  const n = Math.min(days, histFull.fecha.length);
  const hist = {{
    fecha:      histFull.fecha.slice(-n),
    volumen_m3: histFull.volumen_m3.slice(-n),
  }};
  const forecast7 = arima7; // usar ARIMA para mostrar ejemplo en gráfica
  const ctx = document.getElementById('chartHistorico').getContext('2d');
  if (histChart) histChart.destroy();
  histChart = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: [...hist.fecha, ...lstm7.fecha],
      datasets: [
        {{
          label: 'Histórico',
          data: [...hist.volumen_m3.map(v => v/1e6), ...Array(lstm7.fecha.length).fill(null)],
          borderColor: 'rgba(56,189,248,0.8)',
          backgroundColor: 'rgba(56,189,248,0.04)',
          borderWidth: 1.5, pointRadius: 0, fill: true, tension: 0.3,
        }},
        {{
          label: 'LSTM',
          data: [...Array(hist.fecha.length-1).fill(null), hist.volumen_m3[hist.volumen_m3.length-1]/1e6, ...lstm7.volumen_predicho_m3.map(v=>v/1e6)],
          borderColor:'rgba(52,211,153,0.9)', borderWidth:2, borderDash:[4,3], pointRadius:3, pointBackgroundColor:'var(--lstm)', tension:0.3,
        }},
        {{
          label: 'ARIMA',
          data: [...Array(hist.fecha.length-1).fill(null), hist.volumen_m3[hist.volumen_m3.length-1]/1e6, ...arima7.volumen_predicho_m3.map(v=>v/1e6)],
          borderColor:'rgba(245,158,11,0.9)', borderWidth:2, borderDash:[4,3], pointRadius:3, pointBackgroundColor:'var(--arima)', tension:0.3,
        }},
        {{
          label: 'GARCH',
          data: [...Array(hist.fecha.length-1).fill(null), hist.volumen_m3[hist.volumen_m3.length-1]/1e6, ...garch7.volumen_predicho_m3.map(v=>v/1e6)],
          borderColor:'rgba(167,139,250,0.9)', borderWidth:2, borderDash:[4,3], pointRadius:3, pointBackgroundColor:'var(--garch)', tension:0.3,
        }},
        {{
          label: 'Holt-Winters',
          data: [...Array(hist.fecha.length-1).fill(null), hist.volumen_m3[hist.volumen_m3.length-1]/1e6, ...hw7.volumen_predicho_m3.map(v=>v/1e6)],
          borderColor:'rgba(251,113,133,0.9)', borderWidth:2, borderDash:[4,3], pointRadius:3, pointBackgroundColor:'var(--hw)', tension:0.3,
        }},
      ]
    }},
    options: {{
      responsive:true,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{ legend:{{display:false}}, tooltip:tooltipDefaults }},
      scales:{{
        x:{{ticks:{{maxTicksLimit:8,maxRotation:0}}}},
        y:{{title:{{display:true,text:'Volumen (Mm³)'}}}}
      }}
    }}
  }});
}}

function setRange(days, btn) {{
  document.querySelectorAll('.controls .tab').forEach(t => {{
    if(['1S','1M','1A','5A'].some(l => t.textContent===l)) t.classList.remove('active');
  }});
  btn.classList.add('active');
  currentRange = days;
  buildHistChart(days);
}}

buildHistChart(365);

// ── Gráfica IC por modelo ─────────────────────────────────────────────────────
let icChart = null;
let currentModel = 'arima';
let currentICHorizon = 7;

const icData = {{
  arima: {{ 7: arima7,  15: arima15,  30: arima30,  color: 'rgba(245,158,11,0.9)',  bg: 'rgba(245,158,11,0.12)',  label: 'ARIMA' }},
  garch: {{ 7: garch7,  15: garch15,  30: garch30,  color: 'rgba(167,139,250,0.9)', bg: 'rgba(167,139,250,0.12)', label: 'GARCH' }},
  hw:    {{ 7: hw7,     15: hw15,     30: hw30,     color: 'rgba(251,113,133,0.9)', bg: 'rgba(251,113,133,0.12)', label: 'Holt-Winters' }},
  lstm:  {{ 7: lstm7,   15: lstm15,   30: lstm30,   color: 'rgba(52,211,153,0.9)',  bg: 'rgba(52,211,153,0.12)',  label: 'LSTM' }},
}};

function buildICChart(model, h) {{
  const m = icData[model];
  const d = m[h];
  const ctx = document.getElementById('chartIC').getContext('2d');
  if (icChart) icChart.destroy();
  icChart = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: d.fecha,
      datasets: [
        {{
          label: 'IC superior 95%',
          data: d.ic_upper_95.map(v => v/1e6),
          borderColor: 'transparent',
          backgroundColor: m.bg,
          fill: '+1', pointRadius: 0,
        }},
        {{
          label: 'IC inferior 95%',
          data: d.ic_lower_95.map(v => v/1e6),
          borderColor: 'transparent',
          backgroundColor: m.bg,
          fill: false, pointRadius: 0,
        }},
        {{
          label: m.label,
          data: d.volumen_predicho_m3.map(v => v/1e6),
          borderColor: m.color,
          backgroundColor: m.bg,
          borderWidth: 2, pointRadius: 4,
          pointBackgroundColor: m.color,
          tension: 0.3, fill: false,
        }},
      ]
    }},
    options: {{
      responsive:true,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{ legend:{{display:false}}, tooltip:tooltipDefaults }},
      scales:{{
        x:{{ticks:{{maxRotation:0}}}},
        y:{{title:{{display:true,text:'Volumen (Mm³)'}}}}
      }}
    }}
  }});
}}

buildICChart('arima', 7);

function switchModel(model, btn) {{
  document.querySelectorAll('.tab-model').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  currentModel = model;
  buildICChart(model, currentICHorizon);
}}
function switchIC(h, btn) {{
  document.querySelectorAll('.tab-ic').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  currentICHorizon = h;
  buildICChart(currentModel, h);
}}

// ── Gráfica comparativa ───────────────────────────────────────────────────────
let compChart = null;

function buildCompChart(h) {{
  const maps = {{
    7:  {{ arima:arima7,  garch:garch7,  hw:hw7,  lstm:lstm7  }},
    15: {{ arima:arima15, garch:garch15, hw:hw15, lstm:lstm15 }},
    30: {{ arima:arima30, garch:garch30, hw:hw30, lstm:lstm30 }},
  }};
  const d = maps[h];
  const ctx = document.getElementById('chartComparativo').getContext('2d');
  if (compChart) compChart.destroy();
  compChart = new Chart(ctx, {{
    type: 'line',
    data: {{
      labels: d.arima.fecha,
      datasets: [
        {{ label:'ARIMA',        data:d.arima.volumen_predicho_m3.map(v=>v/1e6), borderColor:'rgba(245,158,11,0.9)',  borderWidth:2, pointRadius:3, tension:0.3 }},
        {{ label:'GARCH',        data:d.garch.volumen_predicho_m3.map(v=>v/1e6), borderColor:'rgba(167,139,250,0.9)', borderWidth:2, pointRadius:3, tension:0.3 }},
        {{ label:'Holt-Winters', data:d.hw.volumen_predicho_m3.map(v=>v/1e6),    borderColor:'rgba(251,113,133,0.9)', borderWidth:2, pointRadius:3, tension:0.3 }},
        {{ label:'LSTM',         data:d.lstm.volumen_predicho_m3.map(v=>v/1e6),  borderColor:'rgba(52,211,153,0.9)',  borderWidth:2, pointRadius:3, tension:0.3 }},
      ]
    }},
    options: {{
      responsive:true,
      interaction:{{mode:'index',intersect:false}},
      plugins:{{
        legend:{{display:true,position:'bottom',labels:{{boxWidth:10,font:{{size:10}}}}}},
        tooltip:tooltipDefaults
      }},
      scales:{{
        x:{{ticks:{{maxRotation:0}}}},
        y:{{title:{{display:true,text:'Volumen (Mm³)'}}}}
      }}
    }}
  }});
}}
buildCompChart(7);

function switchComp(h, btn) {{
  document.querySelectorAll('.tab-comp').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  buildCompChart(h);
}}

// ── Métricas ──────────────────────────────────────────────────────────────────
function metRow(n, v) {{ return `<div class="metric-row"><span class="metric-name">${{n}}</span><span class="metric-value">${{v}}</span></div>`; }}
function fmt2(v)   {{ return v != null ? (v/1e6).toFixed(2) + ' Mm³' : 'N/D'; }}
function fmtPct(v) {{ return v != null ? v.toFixed(2)   + '%'   : 'N/D'; }}

function buildMetCard(title, tag, rows) {{
  return `<div class="metrics-card">
    <h3>${{title}} <span class="model-tag tag-${{tag}}">${{tag.toUpperCase()}}</span></h3>
    ${{rows.map(([n,v]) => metRow(n,v)).join('')}}
  </div>`;
}}

document.getElementById('metrics-grid').innerHTML =
  buildMetCard('ARIMA', 'arima', [
    ['MAE test',   fmt2(metARIMA.mae_test)],
    ['RMSE test',  fmt2(metARIMA.rmse_test)],
    ['MAPE test',  fmtPct(metARIMA.mape_test)],
    ['MAE train',  fmt2(metARIMA.mae_train)],
    ['MAPE train', fmtPct(metARIMA.mape_train)],
  ]) +
  buildMetCard('ARMA-GARCH(1,1)', 'garch', [
    ['MAE test',   fmt2(metGARCH.mae_test)],
    ['RMSE test',  fmt2(metGARCH.rmse_test)],
    ['MAPE test',  fmtPct(metGARCH.mape_test)],
    ['MAE train',  fmt2(metGARCH.mae_train)],
    ['MAPE train', fmtPct(metGARCH.mape_train)],
  ]) +
  buildMetCard('Holt-Winters', 'hw', [
    ['MAE test',   fmt2(metHW.mae_test)],
    ['RMSE test',  fmt2(metHW.rmse_test)],
    ['MAPE test',  fmtPct(metHW.mape_test)],
    ['MAE train',  fmt2(metHW.mae_train)],
    ['MAPE train', fmtPct(metHW.mape_train)],
  ]) +
  buildMetCard('LSTM', 'lstm', [
    ['MAE test',   fmt2(metLSTM.mae_test)],
    ['RMSE test',  fmt2(metLSTM.rmse_test)],
    ['MAPE test',  fmtPct(metLSTM.mape_test)],
    ['MAE train',  fmt2(metLSTM.mae_train)],
    ['MAPE train', fmtPct(metLSTM.mape_train)],
  ]);

</script>
</body>
</html>"""

# ── Subir a S3 ────────────────────────────────────────────────────────────────
print('[INFO] Subiendo dashboard a S3...')
s3.put_object(
    Bucket=DASHBOARD_BUCKET,
    Key=DASHBOARD_KEY,
    Body=html.encode('utf-8'),
    ContentType='text/html; charset=utf-8',
)
try:
    s3.put_bucket_website(
        Bucket=DASHBOARD_BUCKET,
        WebsiteConfiguration={{'IndexDocument': {{'Suffix': 'index.html'}}}},
    )
except Exception as e:
    print(f'[WARN] put_bucket_website: {{e}}')

url = 'http://pi-2026-dashboard-387623072987.s3-website-us-east-1.amazonaws.com/index.html'
print(f'[OK] Dashboard disponible en:\n     {{url}}')
print(f'[DONE] {{datetime.now(timezone.utc).isoformat()}}')
