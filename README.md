# Severe Weather & Grid Stress (2024)
**U.S. storm impact on electrical grid performance • 2024**

A Python-only interactive dashboard that links **NOAA Storm Events** to **EIA-930 grid forecast-miss metrics** to help explore **where/when storms intensify** and whether **grid stress patterns** align around those periods (**exploratory association, not causality**).

---

## Demo (what to show in class)
1. **Overview:** animate months → switch storm metric → hover to see NOAA + EIA overlay  
2. **Drilldown:** choose a region → dual-axis timeline → toggle spike markers → small multiples across regions  
3. **Distributions:** CCDF-style view (log x-axis) to show heavy-tailed extremes  
4. **ML Insights:** Ridge lag sensitivity → anomaly days (Isolation Forest) → region×lag heatmap  

---

## Features
### Overview (maps)
- Animated **choropleth** (state-month)
- **Prism spikes** map (discrete bins + “elevation” encoding)
- Optional **EIA overlay** in tooltips (mapped via region buckets)
- “Story beats” to jump to extreme periods

### Drilldown (timeline + regional patterns)
- Dual-axis timeline: **storm metric vs grid metric**
- Zoom via range slider
- Optional spike markers (storm-count and severity spikes)
- Bottom **small multiples**: monthly storms (solid) vs grid anomaly (dashed) across all regions

### Distributions
- CCDF-style “extremes” curves
- **Log x-axis** to emphasize heavy tails
- Supports: storm count, damage, fatalities, severity

### ML Insights (exploratory)
- **Ridge regression** for lag sensitivity (0–7 day lags)
- **Isolation Forest** to surface unusual storm+grid days
- **Region × lag heatmap** to compare timing patterns across regions

---

## Datasets
- **NOAA Storm Events (NCEI):** event records + impacts (damage, injuries, deaths)
- **EIA-930 Hourly Electric Grid Monitor:** demand (MW) and demand forecast (MW)

> ⚠️ Note: EIA forecast miss is a **proxy for operational stress**, not an outage dataset.

---

## Metric definitions (used throughout)
### Grid metrics (EIA)
- **Demand Anomaly (MW)** = Demand (MW) − Demand Forecast (MW)
- **mean_abs_anomaly_mw** = mean(|Demand Anomaly|) → typical forecast miss magnitude
- **max_abs_anomaly_mw** = max(|Demand Anomaly|) → extreme miss magnitude
- **signed_mean_anomaly_mw** = mean(Demand Anomaly) → directional bias

### Storm severity (NOAA)
A unitless composite score:
```
severity = log(1 + property_damage + crop_damage) + 5*deaths + 1*injuries
```
(log compresses heavy-tailed damage; fatalities weighted higher than injuries)

---

## Project structure
```
app/
  app.py                    # Dash entrypoint + navigation
  pages/
    1_overview.py
    2_drilldown.py
    3_distributions.py
    4_ml_insights.py
assets/
  styles.css                # dashboard theme

scripts/
  process_data.py           # build processed artifacts from raw NOAA + EIA

data/
  raw/                      # input CSVs (not committed if large)
  processed/                # generated .pkl files used by the app
```

---

## Setup (local)
### 1) Create and activate a virtual environment
**Windows**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

---

## Data preparation
Place raw files in:
```
data/raw/
  StormEvents_details_*.csv
  StormEvents_fatalities_*.csv
  EIA930_BALANCE_2024_*.csv
  state_to_region.csv
```

Run the processing script (example):
```bash
python scripts/process_data.py --year 2024 \
  --storms_details "data/raw/StormEvents_details_*.csv" \
  --storms_fatalities "data/raw/StormEvents_fatalities_*.csv" \
  --state_region_map "data/raw/state_to_region.csv"
```

Expected outputs (examples):
```
data/processed/state_month.pkl
data/processed/event_type_month.pkl
data/processed/outliers_state_day.pkl
data/processed/storms_region_daily.pkl
data/processed/eia_custom_region_daily.pkl
```

---

## Run the dashboard
```bash
python app/app.py
```

Open:
- http://127.0.0.1:8050

---

## Reproducibility notes / limitations
- This project focuses on **2024**.
- NOAA impacts are subject to reporting uncertainty/bias.
- EIA anomaly metrics are forecast-miss proxies, not direct outage indicators.
- Region mapping is a defendable approximation to align datasets.
- All ML results are **association-based (not causal)**.

---

## Team
- **Samir Sanyal**
- **Rajat Sawant**  
(Indiana University Bloomington)

---

## References
- NOAA NCEI Storm Events Database: https://www.ncei.noaa.gov/access/search/data-search/storm-events  
- EIA-930 Hourly Electric Grid Monitor: https://www.eia.gov/realtime_grid/  

---


