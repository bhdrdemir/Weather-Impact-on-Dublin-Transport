"""
pipeline.py — Data Acquisition, Feature Extraction & Preprocessing Pipeline
============================================================================
Weather Impact on Dublin Public Transport
B9AI001 Programming for Data Analytics — CA2

This module contains all ETL pipeline functions:
  1. Data Acquisition  (API calls + CSV loading)
  2. Feature Extraction (derive meaningful features from raw data)
  3. Transformations    (clean, validate, merge)
  4. Database Loading   (SQLite)
"""

import pandas as pd
import numpy as np
import requests
import sqlite3
import os
import time
import logging
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dublin_transport.db')

SEASON_MAP = {
    12: 'Winter', 1: 'Winter',  2: 'Winter',
    3:  'Spring', 4: 'Spring',  5: 'Spring',
    6:  'Summer', 7: 'Summer',  8: 'Summer',
    9:  'Autumn', 10: 'Autumn', 11: 'Autumn'
}

RAIN_THRESHOLDS = {'dry': 0.2, 'light': 5.0, 'moderate': 15.0, 'heavy': 30.0}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ===================================================================
# SECTION 1: DATA ACQUISITION
# ===================================================================

def fetch_openweather_current(api_key, city='Dublin,IE', units='metric'):
    """
    Fetch current weather from OpenWeatherMap API.

    Parameters
    ----------
    api_key : str — OpenWeatherMap API key (free tier)
    city : str — city and country code
    units : str — 'metric' for Celsius

    Returns
    -------
    dict or None — parsed weather record
    """
    url = 'https://api.openweathermap.org/data/2.5/weather'
    params = {'q': city, 'appid': api_key, 'units': units}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        d = resp.json()
        record = {
            'timestamp': datetime.utcfromtimestamp(d['dt']).isoformat(),
            'temp': d['main']['temp'],
            'temp_min': d['main']['temp_min'],
            'temp_max': d['main']['temp_max'],
            'humidity': d['main']['humidity'],
            'pressure': d['main']['pressure'],
            'wind_speed': d['wind']['speed'],
            'description': d['weather'][0]['description'],
            'rain_1h': d.get('rain', {}).get('1h', 0.0),
            'clouds': d['clouds']['all'],
            'source': 'openweathermap_api'
        }
        logger.info(f"OpenWeatherMap: {record['temp']}°C, {record['description']}")
        return record
    except requests.exceptions.RequestException as e:
        logger.warning(f"OpenWeatherMap API error: {e}")
        return None


def fetch_open_meteo_historical(start_date, end_date,
                                latitude=53.3498, longitude=-6.2603):
    """
    Fetch historical hourly weather from Open-Meteo (free, no key required).

    Parameters
    ----------
    start_date, end_date : str — YYYY-MM-DD format
    latitude, longitude : float — Dublin city centre coordinates

    Returns
    -------
    pd.DataFrame — hourly weather records
    """
    url = 'https://archive-api.open-meteo.com/v1/archive'
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,relative_humidity_2m,rain,wind_speed_10m,'
                  'surface_pressure,cloud_cover',
        'timezone': 'Europe/Dublin'
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hourly = data['hourly']

        df = pd.DataFrame({
            'timestamp': pd.to_datetime(hourly['time']),
            'temp_c': hourly['temperature_2m'],
            'humidity': hourly['relative_humidity_2m'],
            'rain_mm': hourly['rain'],
            'wind_speed_kmh': hourly['wind_speed_10m'],
            'pressure_hpa': hourly['surface_pressure'],
            'cloud_cover': hourly['cloud_cover']
        })
        df['source'] = 'open_meteo_api'
        logger.info(f"Open-Meteo: {len(df)} hourly records ({start_date} to {end_date})")
        return df
    except requests.exceptions.RequestException as e:
        logger.warning(f"Open-Meteo API error: {e}")
        return pd.DataFrame()


def load_met_eireann_daily():
    """Load Met Éireann daily weather CSV (Dublin Airport station)."""
    path = os.path.join(DATA_DIR, 'met_eireann_dublin_daily.csv')
    df = pd.read_csv(path, skiprows=24)
    df.columns = df.columns.str.strip()
    df['date'] = pd.to_datetime(df['date'], format='%d-%b-%Y')

    for col in ['maxtp', 'mintp', 'rain', 'wdsp', 'sun']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

    logger.info(f"Met Éireann daily: {len(df)} records loaded")
    return df


def load_met_eireann_monthly():
    """Load Met Éireann monthly weather CSV."""
    path = os.path.join(DATA_DIR, 'met_eireann_dublin_monthly.csv')
    df = pd.read_csv(path, skiprows=19)
    df.columns = df.columns.str.strip()

    for col in ['meant', 'maxtp', 'mintp', 'mnmax', 'mnmin', 'rain', 'wdsp', 'sun']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')

    df['year'] = df['year'].astype(int)
    df['month'] = df['month'].astype(int)
    logger.info(f"Met Éireann monthly: {len(df)} records loaded")
    return df


def load_dublin_bus():
    """Load Dublin Bus monthly passengers from CSO (Table TOA14)."""
    path = os.path.join(DATA_DIR, 'dublin_bus_monthly_passengers.csv')
    df = pd.read_csv(path)
    df = df[df['Month'] != 'All months'].copy()
    df['passengers'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df['year'] = df['Year'].astype(int)
    df['month'] = df['C01885V02316'].astype(int)
    logger.info(f"Dublin Bus: {len(df)} monthly records loaded")
    return df


def load_luas():
    """Load Luas monthly passengers from CSO (Table TOA11)."""
    path = os.path.join(DATA_DIR, 'luas_passenger_numbers.csv')
    df = pd.read_csv(path)
    df = df[df['Month'] != 'All months'].copy()
    df['passengers'] = pd.to_numeric(df['VALUE'], errors='coerce')
    df['year'] = df['Year'].astype(int)
    df['month'] = df['C01885V02316'].astype(int)

    # Combine Red + Green lines
    total = df.groupby(['year', 'month'], as_index=False)['passengers'].sum()
    logger.info(f"Luas: {len(total)} monthly records loaded (combined lines)")
    return total


def load_weekly_journeys():
    """Load weekly passenger journeys across all modes (CSO Table THA25)."""
    path = os.path.join(DATA_DIR, 'weekly_passenger_journeys.csv')
    df = pd.read_csv(path)
    df['passengers'] = pd.to_numeric(df['VALUE'], errors='coerce')
    logger.info(f"Weekly journeys: {len(df)} records loaded")
    return df


# ===================================================================
# SECTION 2: FEATURE EXTRACTION
# ===================================================================

def extract_temporal_features(df, date_col='date'):
    """
    Extract time-based features from a date column.

    Extracted features:
      - year, month, day, weekday (name), day_of_week (0=Mon)
      - is_weekend (bool)
      - season (meteorological)
      - quarter (Q1-Q4)
      - month_name

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str — column with datetime values

    Returns
    -------
    pd.DataFrame — with new temporal feature columns added
    """
    df = df.copy()
    dt = pd.to_datetime(df[date_col])
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['weekday'] = dt.dt.day_name()
    df['day_of_week'] = dt.dt.dayofweek
    df['is_weekend'] = dt.dt.dayofweek >= 5
    df['season'] = df['month'].map(SEASON_MAP)
    df['quarter'] = dt.dt.quarter
    df['month_name'] = dt.dt.month_name()
    return df


def extract_weather_features(df):
    """
    Extract derived weather features from raw meteorological data.

    Extracted features:
      - avg_temp: mean of daily max and min
      - temp_range: difference between max and min (diurnal range)
      - rain_category: categorical rainfall classification
      - is_rainy: boolean flag (rain > 0.2mm)
      - wind_chill: apparent temperature considering wind
      - weather_severity: composite index (0-10)
      - comfort_index: inverse of severity — higher = more comfortable

    Parameters
    ----------
    df : pd.DataFrame — must have maxtp, mintp, rain, wdsp columns

    Returns
    -------
    pd.DataFrame — with new weather feature columns
    """
    df = df.copy()

    # Average temperature
    if 'maxtp' in df.columns and 'mintp' in df.columns:
        df['avg_temp'] = (df['maxtp'] + df['mintp']) / 2
        df['temp_range'] = df['maxtp'] - df['mintp']

    # Rainfall classification
    if 'rain' in df.columns:
        df['rain_category'] = df['rain'].apply(classify_rainfall)
        df['is_rainy'] = df['rain'] > 0.2

    # Wind chill (simplified Siple formula for moderate conditions)
    if 'avg_temp' in df.columns and 'wdsp' in df.columns:
        wind_kmh = df['wdsp'] * 1.852  # knots to km/h
        df['wind_chill'] = np.where(
            (df['avg_temp'] <= 10) & (wind_kmh > 4.8),
            13.12 + 0.6215 * df['avg_temp']
            - 11.37 * (wind_kmh ** 0.16)
            + 0.3965 * df['avg_temp'] * (wind_kmh ** 0.16),
            df['avg_temp']
        )

    # Weather severity composite index
    if all(c in df.columns for c in ['avg_temp', 'rain', 'wdsp']):
        df['weather_severity'] = df.apply(
            lambda r: compute_weather_severity(
                r.get('avg_temp', np.nan),
                r.get('rain', np.nan),
                r.get('wdsp', np.nan) * 1.852 if pd.notna(r.get('wdsp')) else np.nan
            ), axis=1
        )
        df['comfort_index'] = 10 - df['weather_severity']

    return df


def extract_transport_features(bus_df, luas_df):
    """
    Extract derived features from transport data.

    Extracted features:
      - passengers_norm: min-max normalised (0-1)
      - yoy_change: year-over-year percentage change
      - rolling_avg_3m: 3-month rolling average
      - is_covid: flag for 2020-2021 lockdown period
      - season: meteorological season

    Parameters
    ----------
    bus_df, luas_df : pd.DataFrame — with year, month, passengers columns

    Returns
    -------
    tuple(pd.DataFrame, pd.DataFrame) — enriched bus and luas DataFrames
    """
    for df in [bus_df, luas_df]:
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01')
        df['season'] = df['month'].map(SEASON_MAP)
        df['is_covid'] = df['year'].isin([2020, 2021])

        # Normalise
        df['passengers_norm'] = normalise_passengers(df['passengers'], 'minmax')

        # Year-over-year change
        df = df.sort_values('date')
        df['yoy_change'] = df.groupby('month')['passengers'].pct_change() * 100

        # Rolling average (3 months)
        df = df.sort_values('date')
        df['rolling_avg_3m'] = df['passengers'].rolling(window=3, min_periods=1).mean()

    return bus_df, luas_df


def extract_merged_features(merged_df):
    """
    Extract interaction features from the merged weather+transport dataset.

    Extracted features:
      - rain_group: tercile classification (low/mid/high)
      - temp_group: tercile classification (cold/mild/warm)
      - total_passengers: bus + luas combined
      - bus_share: bus proportion of total
      - weather_impact_score: passengers deviation from seasonal mean

    Parameters
    ----------
    merged_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame — with interaction features
    """
    df = merged_df.copy()

    # Weather tercile groups
    df['rain_group'] = pd.qcut(df['rain'], 3, labels=['low rain', 'mid rain', 'high rain'])
    df['temp_group'] = pd.qcut(df['meant'], 3, labels=['cold', 'mild', 'warm'])

    # Combined transport features
    if 'bus_passengers' in df.columns and 'luas_passengers' in df.columns:
        df['total_passengers'] = df['bus_passengers'] + df['luas_passengers']
        df['bus_share'] = df['bus_passengers'] / df['total_passengers']

    # Weather impact score: how much ridership deviates from seasonal average
    if 'bus_passengers' in df.columns and 'season' in df.columns:
        seasonal_avg = df.groupby('season')['bus_passengers'].transform('mean')
        df['weather_impact_score'] = (
            (df['bus_passengers'] - seasonal_avg) / seasonal_avg * 100
        ).round(2)

    return df


# ===================================================================
# SECTION 3: TRANSFORMATION HELPER FUNCTIONS
# ===================================================================

def classify_rainfall(rain_mm):
    """
    Classify rainfall into categories using Met Éireann thresholds.

    Categories: dry (≤0.2mm), light (≤5mm), moderate (≤15mm),
                heavy (≤30mm), very_heavy (>30mm)
    """
    if pd.isna(rain_mm) or rain_mm < 0:
        return 'unknown'
    if rain_mm <= 0.2:
        return 'dry'
    elif rain_mm <= 5.0:
        return 'light'
    elif rain_mm <= 15.0:
        return 'moderate'
    elif rain_mm <= 30.0:
        return 'heavy'
    return 'very_heavy'


def map_season(month):
    """Map month number (1-12) to meteorological season name."""
    return SEASON_MAP.get(month, 'unknown')


def compute_weather_severity(temp_c, rain_mm, wind_kmh):
    """
    Composite weather severity index (0-10 scale).

    Components (weighted):
      - Cold stress (25%): deviation below 10°C comfort baseline
      - Rain intensity (50%): normalised to 30mm cap
      - Wind severity (25%): normalised to 60 km/h cap

    Higher values = worse conditions for transport usage.
    """
    if any(pd.isna(x) for x in [temp_c, rain_mm, wind_kmh]):
        return np.nan
    cold = max(0, (10 - temp_c) / 20)
    rain = min(rain_mm / 30.0, 1.0)
    wind = min(wind_kmh / 60.0, 1.0)
    return round(cold * 2.5 + rain * 5.0 + wind * 2.5, 2)


def normalise_passengers(passengers, method='minmax'):
    """Normalise passenger counts. Methods: 'minmax' (0-1) or 'zscore'."""
    if method == 'minmax':
        pmin, pmax = passengers.min(), passengers.max()
        if pmax == pmin:
            return pd.Series(0.0, index=passengers.index)
        return (passengers - pmin) / (pmax - pmin)
    elif method == 'zscore':
        std = passengers.std()
        if std == 0:
            return pd.Series(0.0, index=passengers.index)
        return (passengers - passengers.mean()) / std
    raise ValueError(f'Unknown method: {method}')


def validate_dataframe(df, required_cols, name='DataFrame'):
    """Validate DataFrame has expected columns and is non-empty."""
    results = {
        'name': name,
        'rows': len(df),
        'missing_cols': [c for c in required_cols if c not in df.columns],
        'null_counts': {c: int(df[c].isnull().sum()) for c in required_cols if c in df.columns},
        'passed': True
    }
    if results['missing_cols'] or results['rows'] == 0:
        results['passed'] = False
    return results


# ===================================================================
# SECTION 4: DATABASE LOADING
# ===================================================================

def create_database(db_path=None):
    """Create SQLite database with normalised schema."""
    db_path = db_path or DB_PATH
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE weather_daily (
            date TEXT PRIMARY KEY,
            maxtp REAL, mintp REAL, rain REAL, wdsp REAL, sun REAL,
            avg_temp REAL, temp_range REAL, rain_category TEXT,
            is_rainy INTEGER, wind_chill REAL,
            weather_severity REAL, comfort_index REAL,
            year INTEGER, month INTEGER, day INTEGER,
            weekday TEXT, day_of_week INTEGER, is_weekend INTEGER,
            season TEXT, quarter INTEGER
        );

        CREATE TABLE weather_monthly (
            year INTEGER, month INTEGER,
            meant REAL, maxtp REAL, mintp REAL,
            rain REAL, wdsp REAL, sun REAL,
            season TEXT,
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
            rain REAL, meant REAL, wdsp REAL, sun REAL,
            season TEXT, rain_group TEXT, temp_group TEXT,
            weather_impact_score REAL,
            PRIMARY KEY (year, month)
        );

        CREATE TABLE api_weather_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT, temp REAL, humidity REAL,
            rain_mm REAL, wind_speed REAL,
            description TEXT, source TEXT
        );

        CREATE INDEX idx_wd_ym ON weather_daily(year, month);
        CREATE INDEX idx_wd_season ON weather_daily(season);
        CREATE INDEX idx_bus_season ON bus_passengers(season);
        CREATE INDEX idx_luas_season ON luas_passengers(season);
        CREATE INDEX idx_merged_season ON weather_transport_merged(season);
    """)
    conn.commit()
    logger.info(f"Database created: {db_path}")
    return conn


def load_to_database(conn, weather_daily_df, weather_monthly_df,
                     bus_df, luas_df, merged_df):
    """Load all transformed DataFrames into SQLite."""

    # Weather daily
    wd_cols = [c for c in ['date', 'maxtp', 'mintp', 'rain', 'wdsp', 'sun',
               'avg_temp', 'temp_range', 'rain_category', 'is_rainy', 'wind_chill',
               'weather_severity', 'comfort_index',
               'year', 'month', 'day', 'weekday', 'day_of_week', 'is_weekend',
               'season', 'quarter'] if c in weather_daily_df.columns]
    wd = weather_daily_df[wd_cols].copy()
    wd['date'] = wd['date'].astype(str)
    if 'is_rainy' in wd.columns:
        wd['is_rainy'] = wd['is_rainy'].astype(int)
    if 'is_weekend' in wd.columns:
        wd['is_weekend'] = wd['is_weekend'].astype(int)
    wd.to_sql('weather_daily', conn, if_exists='replace', index=False)

    # Weather monthly
    wm_cols = ['year', 'month', 'meant', 'maxtp', 'mintp', 'rain', 'wdsp', 'sun', 'season']
    wm = weather_monthly_df[[c for c in wm_cols if c in weather_monthly_df.columns]]
    wm.to_sql('weather_monthly', conn, if_exists='replace', index=False)

    # Bus
    bus_cols = ['year', 'month', 'passengers', 'passengers_norm',
                'yoy_change', 'rolling_avg_3m', 'season', 'is_covid']
    bc = bus_df[[c for c in bus_cols if c in bus_df.columns]].copy()
    if 'is_covid' in bc.columns:
        bc['is_covid'] = bc['is_covid'].astype(int)
    bc.to_sql('bus_passengers', conn, if_exists='replace', index=False)

    # Luas
    luas_cols = ['year', 'month', 'passengers', 'passengers_norm',
                 'yoy_change', 'rolling_avg_3m', 'season', 'is_covid']
    lc = luas_df[[c for c in luas_cols if c in luas_df.columns]].copy()
    if 'is_covid' in lc.columns:
        lc['is_covid'] = lc['is_covid'].astype(int)
    lc.to_sql('luas_passengers', conn, if_exists='replace', index=False)

    # Merged
    mc = merged_df.copy()
    for col in ['rain_group', 'temp_group']:
        if col in mc.columns:
            mc[col] = mc[col].astype(str)
    mc.to_sql('weather_transport_merged', conn, if_exists='replace', index=False)

    conn.commit()
    logger.info("All data loaded to database")


# ===================================================================
# SECTION 5: FULL PIPELINE ORCHESTRATION
# ===================================================================

def run_full_pipeline(owm_api_key=None):
    """
    Execute the full ETL pipeline end-to-end.

    Steps:
      1. Acquire data (APIs + CSVs)
      2. Extract features
      3. Transform and merge
      4. Load to SQLite database

    Returns
    -------
    dict — pipeline results with status, record counts, and timing
    """
    results = {'steps': [], 'status': 'running', 'start_time': datetime.now().isoformat()}

    try:
        # --- Step 1: Data Acquisition ---
        step = {'name': 'Data Acquisition', 'status': 'running', 'details': {}}

        # 1a. OpenWeatherMap (optional)
        if owm_api_key and owm_api_key != 'YOUR_API_KEY_HERE':
            current = fetch_openweather_current(owm_api_key)
            step['details']['openweathermap'] = 'success' if current else 'failed'
        else:
            step['details']['openweathermap'] = 'skipped (no API key)'

        # 1b. Open-Meteo historical
        api_frames = []
        for year in range(2018, 2026):
            end = f'{year}-12-31' if year < 2025 else '2025-03-31'
            chunk = fetch_open_meteo_historical(f'{year}-01-01', end)
            if not chunk.empty:
                api_frames.append(chunk)
            time.sleep(0.3)
        weather_api = pd.concat(api_frames, ignore_index=True) if api_frames else pd.DataFrame()
        step['details']['open_meteo'] = f'{len(weather_api)} hourly records'

        # 1c. CSV files
        weather_daily = load_met_eireann_daily()
        weather_monthly = load_met_eireann_monthly()
        bus_raw = load_dublin_bus()
        luas_raw = load_luas()
        weekly = load_weekly_journeys()

        step['details']['met_eireann_daily'] = f'{len(weather_daily)} records'
        step['details']['met_eireann_monthly'] = f'{len(weather_monthly)} records'
        step['details']['dublin_bus'] = f'{len(bus_raw)} records'
        step['details']['luas'] = f'{len(luas_raw)} records'
        step['details']['weekly_journeys'] = f'{len(weekly)} records'
        step['status'] = 'complete'
        results['steps'].append(step)

        # --- Step 2: Feature Extraction ---
        step = {'name': 'Feature Extraction', 'status': 'running', 'details': {}}

        weather_daily = extract_temporal_features(weather_daily, 'date')
        weather_daily = extract_weather_features(weather_daily)
        step['details']['weather_features'] = [
            'avg_temp', 'temp_range', 'rain_category', 'is_rainy',
            'wind_chill', 'weather_severity', 'comfort_index',
            'season', 'quarter', 'is_weekend'
        ]

        bus_raw, luas_raw = extract_transport_features(bus_raw, luas_raw)
        step['details']['transport_features'] = [
            'passengers_norm', 'yoy_change', 'rolling_avg_3m',
            'is_covid', 'season'
        ]

        # API data features
        if not weather_api.empty:
            weather_api['date'] = weather_api['timestamp'].dt.date
            api_daily = weather_api.groupby('date').agg(
                avg_temp=('temp_c', 'mean'),
                max_temp=('temp_c', 'max'),
                min_temp=('temp_c', 'min'),
                total_rain=('rain_mm', 'sum'),
                avg_humidity=('humidity', 'mean'),
                avg_wind=('wind_speed_kmh', 'mean')
            ).reset_index()
            step['details']['api_daily_aggregates'] = f'{len(api_daily)} days'

        step['status'] = 'complete'
        results['steps'].append(step)

        # --- Step 3: Transformations (merge) ---
        step = {'name': 'Transformations', 'status': 'running', 'details': {}}

        weather_monthly['season'] = weather_monthly['month'].map(SEASON_MAP)

        # Merge bus + weather
        bus_weather = bus_raw.merge(
            weather_monthly[['year', 'month', 'rain', 'meant', 'wdsp', 'sun']],
            on=['year', 'month'], how='inner')
        bus_weather_nc = bus_weather[~bus_weather['is_covid']].copy()

        # Merge luas + weather
        luas_weather = luas_raw.merge(
            weather_monthly[['year', 'month', 'rain', 'meant', 'wdsp', 'sun']],
            on=['year', 'month'], how='inner')
        luas_weather_nc = luas_weather[~luas_weather['is_covid']].copy()

        # Build merged table
        merged = bus_weather_nc[['year', 'month', 'passengers', 'rain',
                                  'meant', 'wdsp', 'sun', 'season']].copy()
        merged = merged.rename(columns={'passengers': 'bus_passengers'})
        luas_sub = luas_weather_nc[['year', 'month', 'passengers']].rename(
            columns={'passengers': 'luas_passengers'})
        merged = merged.merge(luas_sub, on=['year', 'month'], how='left')
        merged = extract_merged_features(merged)

        step['details']['bus_weather_merged'] = f'{len(bus_weather_nc)} records (ex-COVID)'
        step['details']['luas_weather_merged'] = f'{len(luas_weather_nc)} records (ex-COVID)'
        step['details']['final_merged'] = f'{len(merged)} records'
        step['details']['extracted_interaction_features'] = [
            'rain_group', 'temp_group', 'total_passengers',
            'bus_share', 'weather_impact_score'
        ]
        step['status'] = 'complete'
        results['steps'].append(step)

        # --- Step 4: Database Loading ---
        step = {'name': 'Database Loading', 'status': 'running', 'details': {}}

        conn = create_database()
        load_to_database(conn, weather_daily, weather_monthly,
                         bus_raw, luas_raw, merged)

        # Verify
        cursor = conn.cursor()
        for table in ['weather_daily', 'weather_monthly', 'bus_passengers',
                      'luas_passengers', 'weather_transport_merged']:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            step['details'][table] = f'{count} rows'

        db_size = os.path.getsize(DB_PATH)
        step['details']['db_size'] = f'{db_size / 1024:.1f} KB'
        step['status'] = 'complete'
        results['steps'].append(step)

        conn.close()

        results['status'] = 'complete'
        results['end_time'] = datetime.now().isoformat()
        logger.info("Pipeline completed successfully")

    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        logger.error(f"Pipeline failed: {e}")

    return results


# ===================================================================
# Run pipeline if executed directly
# ===================================================================
if __name__ == '__main__':
    import json
    result = run_full_pipeline()
    print(json.dumps(result, indent=2))
