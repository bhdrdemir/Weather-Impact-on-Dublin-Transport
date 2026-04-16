# Weather Impact on Dublin Public Transport

B9AI001 Programming for Data Analytics — CA2
Bahadir Demir

## What this project does

This is a Flask web app that looks at how weather affects public transport in Dublin. It pulls data from free public APIs, runs a pipeline to extract features and find patterns, then shows everything on a dashboard.

The main idea: when it rains heavily or gets really cold, do buses and trams get more delayed? I built a system that answers this using real data.

## How to run it

You need Python 3.8+ with these packages:

```
pip install flask pandas numpy scipy requests
```

Then just run:

```
python app.py
```

Open `http://localhost:5000` in your browser. Click "Run Pipeline" on the Pipeline tab to fetch all the data (takes about 10 seconds).

## The APIs I used

All free, no API keys needed:

- **Open-Meteo** — historical hourly weather for Dublin (temperature, rain, wind, etc.) from 2022 to now
- **CSO PxStat** — monthly passenger numbers for Dublin Bus (TOA14) and Luas (TOA11) from Ireland's Central Statistics Office
- **Irish Rail API** — live train positions across the network
- **Luas Forecasting API** — real-time tram arrival times

I wrote the API call functions myself in `pipeline.py`. The CSO API returns data in JSON-stat 2.0 format which was tricky to parse — their value field can be either a list or a dict depending on the dataset, so I had to handle both cases.

## How the pipeline works

1. **Data Acquisition** — Calls the APIs and gets raw data into pandas DataFrames
2. **Feature Extraction** — Creates new columns like `is_weekend`, `is_rush_hour`, `rain_category`, `weather_severity` (a 0-10 composite index based on cold+rain+wind)
3. **Transformations** — Merges weather and transport data on year/month, drops COVID years (2020-2021) so lockdowns don't skew the results
4. **Database Loading** — Saves everything into 8 SQLite tables

## What the dashboard shows

- **Live weather** for 9 Dublin districts (D1, D3, D5, D7, D9, D11, D13, D15, D17) with 24h forecast
- **Live transport** — bus routes per area, Luas trams with ETAs, Irish Rail trains
- **Delay prediction** — uses our correlation findings + current weather to estimate delay risk (0-4 levels)
- **Interactive Dublin map** — click a district to switch weather view
- **Correlation charts** — how rainfall, temperature, wind etc. correlate with transport delays
- **Feature extraction showcase** — sample rows from each pipeline stage
- **45 tests** across 12 test classes

## The delay proxy metric

Since we dont have actual delay data, I used a proxy: the percentage that monthly ridership falls below the seasonal baseline. If bus ridership in a rainy January is 15% below the winter average, thats a 15% delay proxy. The idea is that bad weather causes delays which causes fewer people to travel.

## File structure

- `pipeline.py` — all the API calls, feature extraction, and database loading
- `app.py` — Flask backend with API routes
- `templates/index.html` — the frontend (single page app)
- `tests.py` — 45 unit tests + 1 integration test
- `dublin_transport.db` — SQLite database (created when you run the pipeline)

## AI disclosure

I used Claude (Anthropic) as a coding assistant for parts of this project. Being fully transparent:

- **I wrote**: the API call functions, chose the data sources, designed the delay proxy metric, decided on the analysis approach, and did the data exploration
- **AI helped with**: the frontend HTML/CSS/JS, Flask route handlers, test structure, and debugging (like when CSO changed their API response format)

I reviewed and tested all code before using it. The AI didnt make decisions about what to analyse or how — that was all me.

## Tests

Run the tests from the Tests tab in the app, or:

```
python -m pytest tests.py -v
```

45 tests covering: weather severity calculation, feature extraction, temporal features, transport features, CSO parser, daily/monthly aggregation, merged features, normalisation, and a full integration test.
