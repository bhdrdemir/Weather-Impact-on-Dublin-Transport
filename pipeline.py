"""
pipeline.py — 100% API-Driven Data Acquisition & Preprocessing Pipeline
=========================================================================
Weather Impact on Dublin Public Transport
B9AI001 Programming for Data Analytics — CA2

All data is fetched LIVE from APIs — no CSV files used:
  - Open-Meteo API        → Historical hourly weather (2018-2025)
  - OpenWeatherMap API     → Current Dublin weather (real-time)
  - CSO PxStat REST API    → Dublin Bus + Luas monthly passengers
  - Luas Forecasting API   → Real-time tram arrivals
  - Irish Rail API         → Real-time train positions + delays
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
import os
import time
import tempfile
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, 'dublin_transport.db')

SEASON_MAP = {
    12: 'Winter', 1: 'Winter',  2: 'Winter',
    3:  'Spring', 4: 'Spring',  5: 'Spring',
    6:  'Summer', 7: 'Summer',  8: 'Summer',
    9:  'Autumn', 10: 'Autumn', 11: 'Autumn'
}

# All Luas stops (full list for reference & pipeline)
LUAS_STOPS = {
    'marlborough': 'mar', 'oconnell-upper': 'ocu', 'oconnell-gpo': 'ocg',
    'stephens-green': 'stg', 'harcourt': 'har', 'ranelagh': 'ran',
    'beechwood': 'bee', 'cowper': 'cow', 'dundrum': 'dun',
    'balally': 'bal', 'sandyford': 'san', 'cherrywood': 'che',
    'brides-glen': 'brg', 'the-point': 'tpt', 'spencer-dock': 'spd',
    'connolly': 'con', 'busaras': 'bus', 'abbey-street': 'abb',
    'jervis': 'jer', 'smithfield': 'smi', 'heuston': 'heu',
    'red-cow': 'ric', 'tallaght': 'tal', 'saggart': 'sag',
    'parnell': 'par', 'broadstone': 'bro', 'cabra': 'cab',
    'broombridge': 'brm',
}

# Key stops for fast real-time fetching (busiest city-centre stops)
LUAS_KEY_STOPS = {
    'stephens-green': 'stg', 'oconnell-gpo': 'ocg',
    'connolly': 'con', 'heuston': 'heu',
    'abbey-street': 'abb', 'jervis': 'jer',
    'dundrum': 'dun', 'tallaght': 'tal',
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ===================================================================
# SECTION 1: DATA ACQUISITION (100% API)
# ===================================================================

def fetch_open_meteo_historical(start_date, end_date,
                                latitude=53.3498, longitude=-6.2603):
    """
    Fetch historical hourly weather from Open-Meteo archive API.
    Free, no API key required. CC BY 4.0 licence.

    Returns DataFrame with hourly weather records.
    """
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': latitude, 'longitude': longitude,
        'start_date': start_date, 'end_date': end_date,
        'hourly': 'temperature_2m,relative_humidity_2m,rain,wind_speed_10m,'
                  'surface_pressure,cloud_cover',
        'timezone': 'Europe/Dublin'
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        h = resp.json()['hourly']
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(h['time']),
            'temp_c': h['temperature_2m'],
            'humidity': h['relative_humidity_2m'],
            'rain_mm': h['rain'],
            'wind_speed_kmh': h['wind_speed_10m'],
            'pressure_hpa': h['surface_pressure'],
            'cloud_cover': h['cloud_cover']
        })
        df['source'] = 'open_meteo_api'
        return df
    except Exception as e:
        logger.warning(f"Open-Meteo error: {e}")
        return pd.DataFrame()


def fetch_openweather_current(api_key, city='Dublin,IE', units='metric'):
    """
    Fetch current weather from OpenWeatherMap. Requires free API key.
    CC BY-SA 4.0 licence.
    """
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        return None
    url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {'q': city, 'appid': api_key, 'units': units}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        return {
            'timestamp': datetime.utcfromtimestamp(d['dt']).isoformat(),
            'temp': d['main']['temp'],
            'temp_min': d['main']['temp_min'],
            'temp_max': d['main']['temp_max'],
            'humidity': d['main']['humidity'],
            'pressure': d['main']['pressure'],
            'wind_speed': d['wind']['speed'],
            'wind_deg': d['wind'].get('deg', 0),
            'description': d['weather'][0]['description'],
            'icon': d['weather'][0]['icon'],
            'rain_1h': d.get('rain', {}).get('1h', 0.0),
            'clouds': d['clouds']['all'],
            'source': 'openweathermap_api'
        }
    except Exception as e:
        logger.warning(f"OpenWeatherMap error: {e}")
        return None


def fetch_cso_transport(table_id):
    """
    Fetch transport data from CSO PxStat REST API.
    No API key required. CC BY 4.0 licence.

    table_id: 'TOA14' (Dublin Bus) or 'TOA11' (Luas)
    Returns parsed DataFrame with year, month, passengers.
    """
    url = f'https://ws.cso.ie/public/api.restful/PxStat.Data.Cube_API.ReadDataset/{table_id}/JSON-stat/2.0/en'
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        # Parse JSON-stat 2.0 format (value can be list or dict)
        dims = data['dimension']
        dim_keys = list(data['id'])
        values = data['value']
        sizes = data.get('size') or [len(dims[k]['category']['index']) for k in dim_keys]

        # Normalise to (flat_idx, value) iterable
        if isinstance(values, list):
            pairs = enumerate(values)
        else:
            pairs = ((int(k), v) for k, v in values.items())

        records = []
        for flat_idx, value in pairs:
            if value is None:
                continue
            idx = flat_idx
            indices = []
            for s in reversed(sizes):
                indices.insert(0, idx % s)
                idx //= s

            labels = {}
            for i, k in enumerate(dim_keys):
                cat_index = dims[k]['category']['index']
                cat_labels = dims[k]['category']['label']
                # index can be dict {code: position} or list [code, code, ...]
                if isinstance(cat_index, dict):
                    cat_keys = list(cat_index.keys())
                else:
                    cat_keys = list(cat_index)
                if indices[i] < len(cat_keys):
                    labels[k] = cat_labels[cat_keys[indices[i]]]

            labels['value'] = value
            records.append(labels)

        df = pd.DataFrame(records)
        df['source'] = f'cso_api_{table_id}'
        logger.info(f"CSO {table_id}: {len(df)} records fetched")
        return df
    except Exception as e:
        logger.warning(f"CSO API error ({table_id}): {e}")
        return pd.DataFrame()


def fetch_luas_realtime(stop_code='ran'):
    """
    Fetch real-time Luas tram arrivals from Luas Forecasting API.
    No API key required. XML format.

    Returns list of dicts with tram arrival info.
    """
    url = f'http://luasforecasts.rpa.ie/xml/get.ashx?action=forecast&stop={stop_code}&encrypt=false'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'

        stop_name = root.get('stop', stop_code)
        created = root.get('created', '')

        trams = []
        for direction in root.findall(f'{ns}direction'):
            dir_name = direction.get('name', '')
            for tram in direction.findall(f'{ns}tram'):
                trams.append({
                    'stop': stop_name,
                    'stop_code': stop_code,
                    'direction': dir_name,
                    'destination': tram.get('destination', ''),
                    'due_mins': tram.get('dueMins', ''),
                    'timestamp': created,
                    'source': 'luas_api'
                })

        return trams
    except Exception as e:
        logger.warning(f"Luas API error ({stop_code}): {e}")
        return []


def fetch_luas_all_stops(key_only=False):
    """Fetch real-time data for Luas stops. key_only=True for fast dashboard refresh."""
    stops = LUAS_KEY_STOPS if key_only else LUAS_STOPS
    all_trams = []
    for name, code in stops.items():
        trams = fetch_luas_realtime(code)
        all_trams.extend(trams)
        time.sleep(0.1)  # Brief rate limit
    logger.info(f"Luas: {len(all_trams)} tram arrivals across {len(stops)} stops")
    return all_trams


def fetch_irish_rail_realtime():
    """
    Fetch real-time train positions from Irish Rail API.
    No API key required. XML format.

    Returns DataFrame with current train positions and delays.
    """
    url = 'http://api.irishrail.ie/realtime/realtime.asmx/getCurrentTrainsXML'
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        ns = ''
        if root.tag.startswith('{'):
            ns = root.tag.split('}')[0] + '}'

        trains = []
        for t in root.findall(f'{ns}objTrainPositions'):
            trains.append({
                'code': t.findtext(f'{ns}TrainCode', ''),
                'status': t.findtext(f'{ns}TrainStatus', ''),
                'latitude': float(t.findtext(f'{ns}TrainLatitude', '0')),
                'longitude': float(t.findtext(f'{ns}TrainLongitude', '0')),
                'message': t.findtext(f'{ns}PublicMessage', ''),
                'direction': t.findtext(f'{ns}Direction', ''),
                'date': t.findtext(f'{ns}TrainDate', ''),
                'source': 'irish_rail_api'
            })

        logger.info(f"Irish Rail: {len(trains)} active trains")
        return pd.DataFrame(trains)
    except Exception as e:
        logger.warning(f"Irish Rail API error: {e}")
        return pd.DataFrame()


# ===================================================================
# SECTION 2: FEATURE EXTRACTION
# ===================================================================

def extract_temporal_features(df, date_col='timestamp'):
    """
    Extract time-based features from a datetime column.

    Extracted features:
      year, month, day, hour, weekday, day_of_week,
      is_weekend, season, quarter, month_name, is_rush_hour
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['hour'] = dt.dt.hour
    df['weekday'] = dt.dt.day_name()
    df['day_of_week'] = dt.dt.dayofweek
    df['is_weekend'] = dt.dt.dayofweek >= 5
    df['season'] = df['month'].map(SEASON_MAP)
    df['quarter'] = dt.dt.quarter
    df['month_name'] = dt.dt.month_name()
    # Dublin rush hours: 7-9 AM and 4-7 PM on weekdays
    df['is_rush_hour'] = (~df['is_weekend']) & (
        ((df['hour'] >= 7) & (df['hour'] <= 9)) |
        ((df['hour'] >= 16) & (df['hour'] <= 19))
    )
    return df


def extract_weather_features(df):
    """
    Extract derived weather features from hourly data.

    Extracted features:
      rain_category, is_rainy, wind_chill,
      weather_severity (0-10), comfort_index (0-10),
      temp_category, visibility_proxy
    """
    df = df.copy()

    # Rainfall classification
    df['rain_category'] = df['rain_mm'].apply(classify_rainfall)
    df['is_rainy'] = df['rain_mm'] > 0.2

    # Wind chill (simplified)
    df['wind_chill'] = np.where(
        (df['temp_c'] <= 10) & (df['wind_speed_kmh'] > 4.8),
        13.12 + 0.6215 * df['temp_c']
        - 11.37 * (df['wind_speed_kmh'] ** 0.16)
        + 0.3965 * df['temp_c'] * (df['wind_speed_kmh'] ** 0.16),
        df['temp_c']
    )

    # Weather severity composite index
    df['weather_severity'] = df.apply(
        lambda r: compute_weather_severity(r['temp_c'], r['rain_mm'], r['wind_speed_kmh']),
        axis=1
    )
    df['comfort_index'] = 10 - df['weather_severity']

    # Temperature categories
    df['temp_category'] = pd.cut(
        df['temp_c'],
        bins=[-20, 0, 5, 10, 15, 20, 45],
        labels=['freezing', 'very_cold', 'cold', 'mild', 'warm', 'hot']
    )

    # Visibility proxy (high humidity + rain = poor visibility)
    df['poor_visibility'] = (df['humidity'] > 90) & (df['rain_mm'] > 1.0)

    return df


def extract_daily_aggregates(hourly_df):
    """
    Aggregate hourly weather to daily level with extracted features.

    Extracted features:
      avg/max/min temp, total rain, avg wind, avg humidity,
      rain_hours (count of rainy hours), max_severity,
      dry_streak, rain_category_daily
    """
    daily = hourly_df.groupby(hourly_df['timestamp'].dt.date).agg(
        avg_temp=('temp_c', 'mean'),
        max_temp=('temp_c', 'max'),
        min_temp=('temp_c', 'min'),
        total_rain=('rain_mm', 'sum'),
        avg_wind=('wind_speed_kmh', 'mean'),
        max_wind=('wind_speed_kmh', 'max'),
        avg_humidity=('humidity', 'mean'),
        avg_pressure=('pressure_hpa', 'mean'),
        avg_cloud=('cloud_cover', 'mean'),
        rain_hours=('is_rainy', 'sum'),
        max_severity=('weather_severity', 'max'),
        avg_severity=('weather_severity', 'mean'),
    ).reset_index()
    daily.columns = ['date'] + list(daily.columns[1:])
    daily['date'] = pd.to_datetime(daily['date'])

    # Daily-level features
    daily['temp_range'] = daily['max_temp'] - daily['min_temp']
    daily['rain_category'] = daily['total_rain'].apply(classify_rainfall)
    daily['is_rainy'] = daily['total_rain'] > 0.2
    daily['rain_intensity'] = np.where(
        daily['rain_hours'] > 0,
        daily['total_rain'] / daily['rain_hours'],
        0
    )

    return daily


def extract_monthly_aggregates(daily_df):
    """
    Aggregate daily weather to monthly level for transport correlation.

    Extracted features:
      Monthly means + rainy_days count + severe_days count
    """
    daily_df['year'] = daily_df['date'].dt.year
    daily_df['month'] = daily_df['date'].dt.month

    monthly = daily_df.groupby(['year', 'month']).agg(
        mean_temp=('avg_temp', 'mean'),
        total_rain=('total_rain', 'sum'),
        avg_wind=('avg_wind', 'mean'),
        avg_humidity=('avg_humidity', 'mean'),
        rainy_days=('is_rainy', 'sum'),
        avg_severity=('avg_severity', 'mean'),
        max_severity=('max_severity', 'max'),
        severe_days=('max_severity', lambda x: (x > 6).sum()),
    ).reset_index()

    monthly['season'] = monthly['month'].map(SEASON_MAP)
    return monthly


def extract_transport_features(bus_df, luas_df):
    """
    Extract features from CSO transport data.

    Extracted features:
      passengers_norm, yoy_change, rolling_avg_3m, is_covid, season
    """
    results = {}
    for name, df in [('bus', bus_df), ('luas', luas_df)]:
        d = df.copy()
        d['season'] = d['month'].map(SEASON_MAP)
        d['is_covid'] = d['year'].isin([2020, 2021])
        d['passengers_norm'] = normalise_passengers(d['passengers'], 'minmax')
        d = d.sort_values(['year', 'month'])
        d['yoy_change'] = d.groupby('month')['passengers'].pct_change() * 100
        d['rolling_avg_3m'] = d['passengers'].rolling(window=3, min_periods=1).mean()
        results[name] = d
    return results['bus'], results['luas']


def extract_merged_features(merged_df):
    """
    Extract interaction features from weather+transport merged data.

    Extracted features:
      rain_group, temp_group, total_passengers, bus_share,
      weather_impact_score, severity_group
    """
    df = merged_df.copy()
    if df.empty:
        return df

    # Weather tercile groups (duplicates='drop' handles low-variance data)
    df['rain_group'] = pd.qcut(df['total_rain'], 3,
                                labels=['low rain', 'mid rain', 'high rain'],
                                duplicates='drop')
    df['temp_group'] = pd.qcut(df['mean_temp'], 3,
                                labels=['cold', 'mild', 'warm'],
                                duplicates='drop')
    df['severity_group'] = pd.qcut(df['avg_severity'], 3,
                                    labels=['pleasant', 'moderate', 'harsh'],
                                    duplicates='drop')

    # Combined transport
    if 'bus_passengers' in df.columns and 'luas_passengers' in df.columns:
        df['total_passengers'] = df['bus_passengers'] + df['luas_passengers']
        df['bus_share'] = df['bus_passengers'] / df['total_passengers']

    # Weather impact score
    if 'bus_passengers' in df.columns and 'season' in df.columns:
        seasonal_avg = df.groupby('season')['bus_passengers'].transform('mean')
        df['weather_impact_score'] = (
            (df['bus_passengers'] - seasonal_avg) / seasonal_avg * 100
        ).round(2)

    return df


# ===================================================================
# SECTION 3: TRANSFORMATION HELPERS
# ===================================================================

def classify_rainfall(rain_mm):
    """Classify rainfall: dry/light/moderate/heavy/very_heavy."""
    if pd.isna(rain_mm) or rain_mm < 0:
        return 'unknown'
    if rain_mm <= 0.2: return 'dry'
    if rain_mm <= 5.0: return 'light'
    if rain_mm <= 15.0: return 'moderate'
    if rain_mm <= 30.0: return 'heavy'
    return 'very_heavy'


def map_season(month):
    """Map month (1-12) to meteorological season."""
    return SEASON_MAP.get(month, 'unknown')


def compute_weather_severity(temp_c, rain_mm, wind_kmh):
    """
    Composite severity index (0-10).
    Cold stress 25% + rain 50% + wind 25%.
    """
    if any(pd.isna(x) for x in [temp_c, rain_mm, wind_kmh]):
        return np.nan
    cold = max(0, (10 - temp_c) / 20)
    rain = min(rain_mm / 30.0, 1.0)
    wind = min(wind_kmh / 60.0, 1.0)
    return round(cold * 2.5 + rain * 5.0 + wind * 2.5, 2)


def normalise_passengers(passengers, method='minmax'):
    """Normalise: 'minmax' (0-1) or 'zscore' (mean=0, std=1)."""
    if method == 'minmax':
        pmin, pmax = passengers.min(), passengers.max()
        return pd.Series(0.0, index=passengers.index) if pmax == pmin else (passengers - pmin) / (pmax - pmin)
    elif method == 'zscore':
        std = passengers.std()
        return pd.Series(0.0, index=passengers.index) if std == 0 else (passengers - passengers.mean()) / std
    raise ValueError(f'Unknown method: {method}')


def validate_dataframe(df, required_cols, name='DataFrame'):
    """Validate DataFrame structure and content."""
    results = {
        'name': name, 'rows': len(df),
        'missing_cols': [c for c in required_cols if c not in df.columns],
        'null_counts': {c: int(df[c].isnull().sum()) for c in required_cols if c in df.columns},
        'passed': True
    }
    if results['missing_cols'] or results['rows'] == 0:
        results['passed'] = False
    return results


def parse_cso_bus(raw_df):
    """Parse CSO Dublin Bus API response into clean DataFrame."""
    df = raw_df.copy()
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df = df[df['C01885V02316'] != 'All months'].copy()
    df['passengers'] = pd.to_numeric(df['value'], errors='coerce')
    df['year'] = pd.to_numeric(df['TLIST(A1)'], errors='coerce').astype(int)
    df['month'] = df['C01885V02316'].map(month_map)
    df = df.dropna(subset=['passengers', 'month'])
    df['month'] = df['month'].astype(int)
    return df[['year', 'month', 'passengers']].copy()


def parse_cso_luas(raw_df):
    """Parse CSO Luas API response into clean DataFrame (combined lines)."""
    df = raw_df.copy()
    month_map = {'January': 1, 'February': 2, 'March': 3, 'April': 4,
                 'May': 5, 'June': 6, 'July': 7, 'August': 8,
                 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    # Filter to "All Luas lines" if available, else sum
    if 'STATISTIC' in df.columns:
        if 'All Luas lines' in df['STATISTIC'].values:
            df = df[df['STATISTIC'] == 'All Luas lines'].copy()
        else:
            # Sum Red + Green
            df['passengers'] = pd.to_numeric(df['value'], errors='coerce')
            df['year'] = pd.to_numeric(df['TLIST(A1)'], errors='coerce')
            df['month_name'] = df['C01885V02316']
            df = df[df['month_name'] != 'All months']
            df['month'] = df['month_name'].map(month_map)
            df = df.dropna(subset=['passengers', 'month', 'year'])
            df = df.groupby(['year', 'month'], as_index=False)['passengers'].sum()
            df['year'] = df['year'].astype(int)
            df['month'] = df['month'].astype(int)
            return df[['year', 'month', 'passengers']].copy()

    df['passengers'] = pd.to_numeric(df['value'], errors='coerce')
    df['year'] = pd.to_numeric(df['TLIST(A1)'], errors='coerce').astype(int)
    df['month'] = df['C01885V02316'].map(month_map)
    df = df[df['C01885V02316'] != 'All months']
    df = df.dropna(subset=['passengers', 'month'])
    df['month'] = df['month'].astype(int)
    return df[['year', 'month', 'passengers']].copy()


# ===================================================================
# SECTION 4: DATABASE LOADING
# ===================================================================

def create_database(db_path=None):
    """Create SQLite database with schema."""
    db_path = db_path or DB_PATH
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE weather_hourly (
            timestamp TEXT, temp_c REAL, humidity REAL, rain_mm REAL,
            wind_speed_kmh REAL, pressure_hpa REAL, cloud_cover REAL,
            rain_category TEXT, is_rainy INTEGER, wind_chill REAL,
            weather_severity REAL, comfort_index REAL, temp_category TEXT,
            poor_visibility INTEGER,
            year INTEGER, month INTEGER, day INTEGER, hour INTEGER,
            weekday TEXT, is_weekend INTEGER, season TEXT, is_rush_hour INTEGER,
            source TEXT
        );

        CREATE TABLE weather_daily (
            date TEXT PRIMARY KEY,
            avg_temp REAL, max_temp REAL, min_temp REAL,
            total_rain REAL, avg_wind REAL, max_wind REAL,
            avg_humidity REAL, avg_pressure REAL, avg_cloud REAL,
            rain_hours REAL, max_severity REAL, avg_severity REAL,
            temp_range REAL, rain_category TEXT, is_rainy INTEGER,
            rain_intensity REAL
        );

        CREATE TABLE weather_monthly (
            year INTEGER, month INTEGER,
            mean_temp REAL, total_rain REAL, avg_wind REAL,
            avg_humidity REAL, rainy_days REAL, avg_severity REAL,
            max_severity REAL, severe_days INTEGER, season TEXT,
            PRIMARY KEY (year, month)
        );

        CREATE TABLE bus_passengers (
            year INTEGER, month INTEGER,
            passengers INTEGER, passengers_norm REAL,
            yoy_change REAL, rolling_avg_3m REAL,
            season TEXT, is_covid INTEGER,
            PRIMARY KEY (year, month)
        );

        CREATE TABLE luas_passengers (
            year INTEGER, month INTEGER,
            passengers INTEGER, passengers_norm REAL,
            yoy_change REAL, rolling_avg_3m REAL,
            season TEXT, is_covid INTEGER,
            PRIMARY KEY (year, month)
        );

        CREATE TABLE weather_transport_merged (
            year INTEGER, month INTEGER,
            bus_passengers INTEGER, luas_passengers INTEGER,
            total_passengers INTEGER, bus_share REAL,
            total_rain REAL, mean_temp REAL, avg_wind REAL,
            avg_severity REAL, rainy_days REAL, severe_days INTEGER,
            season TEXT, rain_group TEXT, temp_group TEXT,
            severity_group TEXT, weather_impact_score REAL,
            PRIMARY KEY (year, month)
        );

        CREATE TABLE luas_realtime (
            stop TEXT, stop_code TEXT, direction TEXT,
            destination TEXT, due_mins TEXT, timestamp TEXT
        );

        CREATE TABLE rail_realtime (
            code TEXT, status TEXT, latitude REAL, longitude REAL,
            message TEXT, direction TEXT, date TEXT
        );

        CREATE INDEX idx_wh_ym ON weather_hourly(year, month);
        CREATE INDEX idx_wh_season ON weather_hourly(season);
        CREATE INDEX idx_wd_date ON weather_daily(date);
        CREATE INDEX idx_merged_season ON weather_transport_merged(season);
    """)
    conn.commit()
    return conn


def load_to_database(conn, hourly_df, daily_df, monthly_df,
                     bus_df, luas_df, merged_df,
                     luas_rt=None, rail_rt=None):
    """Load all DataFrames into SQLite."""

    def safe_int(df, cols):
        d = df.copy()
        for c in cols:
            if c in d.columns:
                d[c] = d[c].astype(int)
        return d

    # Hourly weather
    hw = hourly_df.copy()
    for c in ['is_rainy', 'is_weekend', 'is_rush_hour', 'poor_visibility']:
        if c in hw.columns:
            hw[c] = hw[c].astype(int)
    hw['timestamp'] = hw['timestamp'].astype(str)
    if 'temp_category' in hw.columns:
        hw['temp_category'] = hw['temp_category'].astype(str)
    hw.to_sql('weather_hourly', conn, if_exists='replace', index=False)

    # Daily weather
    dd = daily_df.copy()
    dd['date'] = dd['date'].astype(str)
    if 'is_rainy' in dd.columns:
        dd['is_rainy'] = dd['is_rainy'].astype(int)
    dd.to_sql('weather_daily', conn, if_exists='replace', index=False)

    # Monthly weather
    monthly_df.to_sql('weather_monthly', conn, if_exists='replace', index=False)

    # Bus
    bc = safe_int(bus_df, ['is_covid'])
    bc.to_sql('bus_passengers', conn, if_exists='replace', index=False)

    # Luas
    lc = safe_int(luas_df, ['is_covid'])
    lc.to_sql('luas_passengers', conn, if_exists='replace', index=False)

    # Merged
    mc = merged_df.copy()
    for c in ['rain_group', 'temp_group', 'severity_group']:
        if c in mc.columns:
            mc[c] = mc[c].astype(str)
    mc.to_sql('weather_transport_merged', conn, if_exists='replace', index=False)

    # Real-time data
    if luas_rt:
        pd.DataFrame(luas_rt).to_sql('luas_realtime', conn, if_exists='replace', index=False)
    if rail_rt is not None and not rail_rt.empty:
        rail_rt.to_sql('rail_realtime', conn, if_exists='replace', index=False)

    conn.commit()
    logger.info("All data loaded to database")


# ===================================================================
# SECTION 5: PIPELINE ORCHESTRATION
# ===================================================================

def run_full_pipeline(owm_api_key=None, progress_callback=None):
    """
    Full ETL pipeline: Acquire → Extract Features → Transform → Load.
    progress_callback(step_name, status, details) is called at each stage.
    """
    def notify(step, status, details):
        if progress_callback:
            progress_callback(step, status, details)
        logger.info(f"[{step}] {status}: {details}")

    results = {'steps': [], 'status': 'running', 'start_time': datetime.now().isoformat()}

    try:
        # ============ STEP 1: DATA ACQUISITION ============
        step = {'name': 'Data Acquisition', 'status': 'running', 'details': {}}
        notify('Data Acquisition', 'running', 'Fetching from 5 APIs...')

        # 1a. Open-Meteo historical — 2022 through latest archive-available day
        #     Archive has ~5-day lag; go up to 5 days before today
        start_date = '2022-01-01'
        end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        notify('Data Acquisition', 'running', f'Open-Meteo: {start_date} → {end_date}...')
        weather_hourly = fetch_open_meteo_historical(start_date, end_date)
        step['details']['open_meteo'] = f'{len(weather_hourly):,} hourly records'

        # 1b. CSO Dublin Bus passengers (monthly, all years)
        notify('Data Acquisition', 'running', 'CSO: Dublin Bus passengers...')
        bus_raw = fetch_cso_transport('TOA14')
        step['details']['cso_bus'] = f'{len(bus_raw)} records'

        # 1c. CSO Luas passengers (monthly, all years)
        notify('Data Acquisition', 'running', 'CSO: Luas passengers...')
        luas_raw = fetch_cso_transport('TOA11')
        step['details']['cso_luas'] = f'{len(luas_raw)} records'

        # 1d. Irish Rail real-time (fast, single call)
        notify('Data Acquisition', 'running', 'Irish Rail: real-time trains...')
        rail_rt = fetch_irish_rail_realtime()
        step['details']['irish_rail'] = f'{len(rail_rt)} active trains'

        # Luas real-time is handled live by /api/live (not part of pipeline)
        luas_rt = []

        step['status'] = 'complete'
        results['steps'].append(step)

        # ============ STEP 2: FEATURE EXTRACTION ============
        step = {'name': 'Feature Extraction', 'status': 'running', 'details': {}}
        notify('Feature Extraction', 'running', 'Extracting weather features...')

        weather_hourly = extract_temporal_features(weather_hourly, 'timestamp')
        weather_hourly = extract_weather_features(weather_hourly)
        step['details']['weather_features'] = [
            'rain_category', 'is_rainy', 'wind_chill', 'weather_severity',
            'comfort_index', 'temp_category', 'poor_visibility',
            'is_rush_hour', 'is_weekend', 'season'
        ]

        notify('Feature Extraction', 'running', 'Aggregating daily/monthly...')
        weather_daily = extract_daily_aggregates(weather_hourly)
        weather_monthly = extract_monthly_aggregates(weather_daily)
        step['details']['daily_records'] = f'{len(weather_daily):,} days'
        step['details']['monthly_records'] = f'{len(weather_monthly)} months'

        notify('Feature Extraction', 'running', 'Extracting transport features...')
        bus_clean = parse_cso_bus(bus_raw)
        luas_clean = parse_cso_luas(luas_raw)
        bus_feat, luas_feat = extract_transport_features(bus_clean, luas_clean)
        step['details']['transport_features'] = [
            'passengers_norm', 'yoy_change', 'rolling_avg_3m', 'is_covid'
        ]

        step['status'] = 'complete'
        results['steps'].append(step)

        # ============ STEP 3: TRANSFORMATIONS ============
        step = {'name': 'Transformations', 'status': 'running', 'details': {}}
        notify('Transformations', 'running', 'Merging weather + transport...')

        # Merge on year/month (exclude COVID)
        bus_nc = bus_feat[~bus_feat['is_covid']].copy()
        luas_nc = luas_feat[~luas_feat['is_covid']].copy()

        merged = bus_nc[['year', 'month', 'passengers', 'season']].rename(
            columns={'passengers': 'bus_passengers'})
        luas_sub = luas_nc[['year', 'month', 'passengers']].rename(
            columns={'passengers': 'luas_passengers'})
        merged = merged.merge(luas_sub, on=['year', 'month'], how='inner')
        merged = merged.merge(
            weather_monthly[['year', 'month', 'mean_temp', 'total_rain',
                            'avg_wind', 'avg_severity', 'rainy_days', 'severe_days']],
            on=['year', 'month'], how='inner')

        merged = extract_merged_features(merged)
        step['details']['merged_records'] = f'{len(merged)} month-pairs (ex-COVID)'
        step['details']['interaction_features'] = [
            'rain_group', 'temp_group', 'severity_group',
            'total_passengers', 'bus_share', 'weather_impact_score'
        ]

        step['status'] = 'complete'
        results['steps'].append(step)

        # ============ STEP 4: DATABASE LOADING ============
        step = {'name': 'Database Loading', 'status': 'running', 'details': {}}
        notify('Database Loading', 'running', 'Creating SQLite database...')

        conn = create_database()
        load_to_database(conn, weather_hourly, weather_daily, weather_monthly,
                         bus_feat, luas_feat, merged, luas_rt, rail_rt)

        cursor = conn.cursor()
        for table in ['weather_hourly', 'weather_daily', 'weather_monthly',
                      'bus_passengers', 'luas_passengers',
                      'weather_transport_merged', 'luas_realtime', 'rail_realtime']:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            step['details'][table] = f'{count:,} rows'

        db_size = os.path.getsize(DB_PATH)
        step['details']['db_size'] = f'{db_size / (1024*1024):.1f} MB'

        step['status'] = 'complete'
        results['steps'].append(step)
        conn.close()

        results['status'] = 'complete'
        results['end_time'] = datetime.now().isoformat()
        notify('Pipeline', 'complete', 'All steps finished successfully')

    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        logger.error(f"Pipeline failed: {e}", exc_info=True)

    return results


if __name__ == '__main__':
    import json
    result = run_full_pipeline()
    print(json.dumps(result, indent=2))
