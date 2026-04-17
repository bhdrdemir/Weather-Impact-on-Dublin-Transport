# Weather Impact on Dublin Public Transport

B9AI001 Programming for Data Analytics — CA2
Bahadir Demir

## What it does

Flask web app that analyses how weather affects public transport in Dublin. Pulls live data from 4 public APIs, runs a feature extraction pipeline, stores results in SQLite, and shows an interactive dashboard with weather-based delay and crowdedness predictions.

## How to run

```
pip install flask pandas numpy scipy requests
python app.py
```

Open `http://localhost:5000`. Hit "Run Pipeline" on the Pipeline tab to fetch historical data (~10 seconds).

## APIs used

All free, no keys needed:

- **Open-Meteo** — hourly weather for Dublin (2022-now) + live multi-location forecast for 9 districts
- **CSO PxStat** — monthly bus (TOA14) and Luas (TOA11) passenger counts
- **Irish Rail** — real-time train positions (XML)
- **Luas Forecasting** — live tram arrivals per stop (XML)

## Pipeline

1. **Data Acquisition** — fetches from 3 APIs into DataFrames
2. **Feature Extraction** — derives ~25 columns (weather_severity, is_rush_hour, rain_category, etc.)
3. **Transformations** — merges weather + transport on year/month, drops COVID years (2020-2021)
4. **Database Loading** — 8 normalised SQLite tables

## Dashboard

- Live weather for 9 Dublin districts with 24h forecast and weather map
- Bus arrival board with dropdown stop selector — shows ETA, direction, delay and crowdedness predictions per bus
- Delay risk banner using historical correlation coefficients + live weather
- Correlation charts showing weather vs delay impact and crowdedness patterns
- Full feature extraction showcase, database browser, SQL editor
- 45 tests across 12 classes (run from Tests tab)

## Files

- `pipeline.py` — API calls, feature extraction, database loading
- `app.py` — Flask backend with all routes
- `templates/index.html` — single-page frontend
- `tests.py` — 45 unit + integration tests
- `dublin_transport.db` — SQLite (created by pipeline)

## AI disclosure

Used an AI coding assistant for the frontend dashboard (HTML/CSS/JS), test structure, and debugging. I wrote the API integration code, Flask route handlers, chose the data sources, designed the analysis approach, and built the delay proxy metric. All AI output was reviewed and tested before use.
