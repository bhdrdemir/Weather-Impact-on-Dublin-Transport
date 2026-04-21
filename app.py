"""
app.py — Flask Backend for Weather Impact on Dublin Transport Pipeline
======================================================================
B9AI001 Programming for Data Analytics — CA2

All API routes serving the frontend dashboard.
"""

from flask import Flask, render_template, jsonify, request
import sqlite3
import pandas as pd
import numpy as np
import os
import json
import threading
from datetime import datetime
from scipy.stats import pearsonr, spearmanr, kruskal

from pipeline import (
    run_full_pipeline, DB_PATH,
    fetch_openweather_current, fetch_luas_realtime,
    fetch_irish_rail_realtime, fetch_luas_all_stops,
    LUAS_STOPS, LUAS_KEY_STOPS, SEASON_MAP,
    compute_weather_severity
)
from tests import run_tests

app = Flask(__name__)

# Pipeline state
pipeline_state = {
    'status': 'idle',
    'current_step': '',
    'current_detail': '',
    'last_run': None,
    'result': None,
    'log': []
}


def pipeline_progress(step, status, details):
    """Callback for pipeline progress updates."""
    pipeline_state['current_step'] = step
    pipeline_state['current_detail'] = details
    pipeline_state['log'].append({
        'time': datetime.now().strftime('%H:%M:%S'),
        'step': step,
        'status': status,
        'detail': details
    })


def get_db():
    if not os.path.exists(DB_PATH):
        return None
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ===================================================================
# ROUTES
# ===================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/pipeline/run', methods=['POST'])
def api_run_pipeline():
    if pipeline_state['status'] == 'running':
        return jsonify({'error': 'Pipeline already running'}), 409

    api_key = request.json.get('api_key', '') if request.is_json else ''

    def _run():
        pipeline_state['status'] = 'running'
        pipeline_state['result'] = None
        pipeline_state['log'] = []
        try:
            result = run_full_pipeline(
                owm_api_key=api_key or None,
                progress_callback=pipeline_progress
            )
            pipeline_state['result'] = result
            pipeline_state['status'] = result.get('status', 'complete')
        except Exception as e:
            pipeline_state['status'] = 'failed'
            pipeline_state['result'] = {'error': str(e)}
        pipeline_state['last_run'] = datetime.now().isoformat()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({'message': 'Pipeline started'})


@app.route('/api/pipeline/status')
def api_pipeline_status():
    return jsonify({
        'status': pipeline_state['status'],
        'current_step': pipeline_state['current_step'],
        'current_detail': pipeline_state['current_detail'],
        'last_run': pipeline_state['last_run'],
        'result': pipeline_state['result'],
        'log': pipeline_state['log'][-50:]  # Last 50 log entries
    })


@app.route('/api/stats')
def api_stats():
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found. Run pipeline first.'}), 404

    try:
        df = pd.read_sql_query('SELECT * FROM weather_transport_merged', conn)

        # Compute a "delay / disruption proxy" per mode:
        #   delay_proxy = % shortfall of actual passengers vs the mode's baseline
        #   (baseline = that mode's overall mean within the same season).
        # Higher values mean ridership dropped below what's normal for the season —
        # a plausible signature of weather-driven delays/disruption.
        for mode, col in [('bus', 'bus_passengers'), ('luas', 'luas_passengers')]:
            delay_col = f'{mode}_delay_proxy_pct'
            baseline = df.groupby('season')[col].transform('mean')
            df[delay_col] = ((baseline - df[col]) / baseline * 100).clip(lower=0)

        results = {}
        for mode, col in [('bus', 'bus_passengers'), ('luas', 'luas_passengers')]:
            delay_col = f'{mode}_delay_proxy_pct'
            correlations = []
            for var, label in [('total_rain', 'Rainfall (mm)'),
                               ('mean_temp', 'Temperature (°C)'),
                               ('avg_wind', 'Wind Speed (km/h)'),
                               ('rainy_days', 'Rainy Days'),
                               ('avg_severity', 'Weather Severity')]:
                if var in df.columns and delay_col in df.columns:
                    valid = df[[var, delay_col]].dropna()
                    if len(valid) > 5:
                        pr, pp = pearsonr(valid[var], valid[delay_col])
                        sr, sp = spearmanr(valid[var], valid[delay_col])
                        correlations.append({
                            'variable': label,
                            'pearson_r': round(pr, 4), 'pearson_p': round(pp, 4),
                            'spearman_r': round(sr, 4), 'spearman_p': round(sp, 4),
                            'significant': bool(pp < 0.05),
                        })
            results[mode] = correlations

        # Kruskal-Wallis — group delay-proxy by weather category
        kw_tests = []
        for mode, label_mode in [('bus', 'Bus'), ('luas', 'Luas')]:
            delay_col = f'{mode}_delay_proxy_pct'
            for gcol, glabel in [('rain_group', 'Rainfall'),
                                  ('temp_group', 'Temperature'),
                                  ('severity_group', 'Severity')]:
                if gcol in df.columns and delay_col in df.columns:
                    groups = [g[delay_col].dropna().values for _, g in df.groupby(gcol)]
                    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                        stat, p = kruskal(*groups)
                        kw_tests.append({
                            'test': f'{label_mode} delay × {glabel}',
                            'statistic': round(stat, 3),
                            'p_value': round(p, 4),
                            'significant': bool(p < 0.05)
                        })
        results['kruskal_wallis'] = kw_tests

        # Delay samples: max delay-proxy month per weather category
        delay_examples = []
        for label_mode, delay_col in [('Bus', 'bus_delay_proxy_pct'),
                                       ('Luas', 'luas_delay_proxy_pct')]:
            if delay_col in df.columns:
                top = df.nlargest(3, delay_col)[['year', 'month', 'season',
                                                  'total_rain', 'mean_temp',
                                                  'avg_wind', delay_col]]
                for _, row in top.iterrows():
                    delay_examples.append({
                        'mode': label_mode,
                        'period': f"{int(row['year'])}-{int(row['month']):02d}",
                        'season': row['season'],
                        'rain': round(row['total_rain'], 1),
                        'temp': round(row['mean_temp'], 1),
                        'wind': round(row['avg_wind'], 1),
                        'delay_pct': round(row[delay_col], 1),
                    })
        results['delay_examples'] = delay_examples

        # Crowdedness correlations: weather vs actual passenger counts
        # More passengers = more crowded, so positive correlation with rain
        # means "when it rains more, more people take public transport → more crowded"
        crowd_correlations = {}
        for mode, col in [('bus', 'bus_passengers'), ('luas', 'luas_passengers')]:
            corrs = []
            for var, label in [('total_rain', 'Rainfall (mm)'),
                               ('mean_temp', 'Temperature (°C)'),
                               ('avg_wind', 'Wind Speed (km/h)'),
                               ('rainy_days', 'Rainy Days'),
                               ('avg_severity', 'Weather Severity')]:
                if var in df.columns and col in df.columns:
                    valid = df[[var, col]].dropna()
                    if len(valid) > 5:
                        pr, pp = pearsonr(valid[var], valid[col])
                        sr, sp = spearmanr(valid[var], valid[col])
                        corrs.append({
                            'variable': label,
                            'pearson_r': round(pr, 4), 'pearson_p': round(pp, 4),
                            'spearman_r': round(sr, 4), 'spearman_p': round(sp, 4),
                            'significant': bool(pp < 0.05),
                        })
            crowd_correlations[mode] = corrs
        results['crowdedness'] = crowd_correlations

        results['summary'] = {
            'total_records': len(df),
            'year_range': f"{int(df['year'].min())}–{int(df['year'].max())}",
            'avg_bus': int(df['bus_passengers'].mean()) if 'bus_passengers' in df.columns else 0,
            'avg_luas': int(df['luas_passengers'].mean()) if 'luas_passengers' in df.columns else 0,
            'avg_rain': round(df['total_rain'].mean(), 1),
            'avg_temp': round(df['mean_temp'].mean(), 1),
            'avg_severity': round(df['avg_severity'].mean(), 2) if 'avg_severity' in df.columns else 0,
            'avg_bus_delay_pct': round(df['bus_delay_proxy_pct'].mean(), 2),
            'avg_luas_delay_pct': round(df['luas_delay_proxy_pct'].mean(), 2),
        }

        conn.close()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/query')
def api_query():
    sql = request.args.get('sql', '').strip()
    if not sql:
        return jsonify({'error': 'No SQL provided'}), 400
    if not sql.upper().startswith('SELECT'):
        return jsonify({'error': 'Only SELECT queries allowed'}), 403

    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found'}), 404

    try:
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return jsonify({
            'columns': list(df.columns),
            'data': json.loads(df.head(500).to_json(orient='records')),
            'total_rows': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/tables')
def api_tables():
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found'}), 404

    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = []
    for row in cursor.fetchall():
        name = row[0]
        cursor.execute(f'SELECT COUNT(*) FROM "{name}"')
        count = cursor.fetchone()[0]
        cursor.execute(f'PRAGMA table_info("{name}")')
        columns = [col[1] for col in cursor.fetchall()]
        tables.append({'name': name, 'rows': count, 'columns': columns})
    conn.close()
    return jsonify(tables)


@app.route('/api/tests/run')
def api_run_tests():
    try:
        return jsonify(run_tests())
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/weather/current')
def api_current_weather():
    api_key = request.args.get('api_key', '')
    if not api_key:
        return jsonify({'error': 'API key required'}), 400
    result = fetch_openweather_current(api_key)
    return jsonify(result) if result else (jsonify({'error': 'Failed'}), 502)


@app.route('/api/luas/realtime')
def api_luas_rt():
    stop = request.args.get('stop', '')
    if stop:
        return jsonify(fetch_luas_realtime(stop))
    return jsonify(fetch_luas_all_stops())


@app.route('/api/rail/realtime')
def api_rail_rt():
    df = fetch_irish_rail_realtime()
    if df.empty:
        return jsonify([])
    return jsonify(json.loads(df.to_json(orient='records')))


# Dublin postal areas with coordinates (centroid of each area)
DUBLIN_AREAS = {
    'D1':  {'name': 'Dublin 1 — City North', 'lat': 53.3547, 'lon': -6.2555},
    'D3':  {'name': 'Dublin 3 — Clontarf', 'lat': 53.3649, 'lon': -6.2100},
    'D5':  {'name': 'Dublin 5 — Raheny', 'lat': 53.3811, 'lon': -6.1781},
    'D7':  {'name': 'Dublin 7 — Phibsborough', 'lat': 53.3575, 'lon': -6.2729},
    'D9':  {'name': 'Dublin 9 — Drumcondra', 'lat': 53.3760, 'lon': -6.2552},
    'D11': {'name': 'Dublin 11 — Finglas', 'lat': 53.3900, 'lon': -6.2980},
    'D13': {'name': 'Dublin 13 — Donaghmede', 'lat': 53.3940, 'lon': -6.1510},
    'D15': {'name': 'Dublin 15 — Blanchardstown', 'lat': 53.3872, 'lon': -6.3782},
    'D17': {'name': 'Dublin 17 — Coolock', 'lat': 53.3870, 'lon': -6.2010},
}


WMO_CODES = {
    0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
    45: 'Foggy', 48: 'Rime fog', 51: 'Light drizzle',
    53: 'Moderate drizzle', 55: 'Dense drizzle', 56: 'Freezing drizzle',
    57: 'Dense freezing drizzle', 61: 'Slight rain', 63: 'Moderate rain',
    65: 'Heavy rain', 66: 'Light freezing rain', 67: 'Heavy freezing rain',
    71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
    80: 'Slight showers', 81: 'Moderate showers', 82: 'Violent showers',
    95: 'Thunderstorm',
}


@app.route('/api/areas')
def api_areas():
    """List of supported Dublin areas for the weather selector."""
    return jsonify([{'code': k, **v} for k, v in DUBLIN_AREAS.items()])


@app.route('/api/live')
def api_live():
    """
    Live data endpoint — fetches ALL areas in a single Open-Meteo call
    plus Luas/Rail. Frontend caches it and switches areas instantly.
    """
    import requests as _req
    from concurrent.futures import ThreadPoolExecutor

    result = {'timestamp': datetime.now().isoformat(), 'areas': {}}

    # Fetch ALL areas in a SINGLE Open-Meteo call (multi-location)
    lats = ','.join(str(a['lat']) for a in DUBLIN_AREAS.values())
    lons = ','.join(str(a['lon']) for a in DUBLIN_AREAS.values())
    area_codes = list(DUBLIN_AREAS.keys())

    try:
        resp = _req.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': lats, 'longitude': lons,
            'current': 'temperature_2m,relative_humidity_2m,rain,wind_speed_10m,'
                       'surface_pressure,cloud_cover,weather_code',
            'hourly': 'temperature_2m,rain,wind_speed_10m,weather_code',
            'forecast_hours': 24,
            'timezone': 'Europe/Dublin'
        }, timeout=10)
        resp.raise_for_status()
        all_data = resp.json()

        # Multi-location returns a list of results
        if not isinstance(all_data, list):
            all_data = [all_data]

        for i, code in enumerate(area_codes):
            data = all_data[i] if i < len(all_data) else {}
            c = data.get('current', {})
            wcode = c.get('weather_code', 0)
            severity = compute_weather_severity(
                c.get('temperature_2m', 10), c.get('rain', 0), c.get('wind_speed_10m', 10))
            weather = {
                'temp': c.get('temperature_2m'),
                'humidity': c.get('relative_humidity_2m'),
                'rain': c.get('rain', 0),
                'wind_speed': c.get('wind_speed_10m'),
                'pressure': c.get('surface_pressure'),
                'cloud_cover': c.get('cloud_cover'),
                'description': WMO_CODES.get(wcode, f'Code {wcode}'),
                'weather_code': wcode,
                'severity': severity,
            }
            h = data.get('hourly', {})
            times = h.get('time', [])
            temps = h.get('temperature_2m', [])
            rains = h.get('rain', [])
            winds = h.get('wind_speed_10m', [])
            codes = h.get('weather_code', [])
            forecast = []
            for j in range(min(24, len(times))):
                forecast.append({
                    'time': times[j],
                    'temp': temps[j] if j < len(temps) else None,
                    'rain': rains[j] if j < len(rains) else 0,
                    'wind_speed': winds[j] if j < len(winds) else None,
                    'weather_code': codes[j] if j < len(codes) else 0,
                    'description': WMO_CODES.get(codes[j] if j < len(codes) else 0, ''),
                })
            result['areas'][code] = {
                'name': DUBLIN_AREAS[code]['name'],
                'weather': weather, 'forecast': forecast,
            }
    except Exception as e:
        # Fallback: empty weather for all areas
        for code in area_codes:
            result['areas'][code] = {
                'name': DUBLIN_AREAS[code]['name'],
                'weather': {'error': str(e)}, 'forecast': [],
            }

    # Luas + Rail fetched in parallel
    def get_luas():
        try:
            return fetch_luas_all_stops(key_only=True)
        except Exception:
            return []

    def get_rail():
        try:
            df = fetch_irish_rail_realtime()
            return json.loads(df.to_json(orient='records')) if not df.empty else []
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=2) as pool:
        luas_future = pool.submit(get_luas)
        rail_future = pool.submit(get_rail)
        luas = luas_future.result()
        rail = rail_future.result()

    result['luas'] = luas
    result['luas_count'] = len(luas)
    result['rail'] = rail
    result['rail_count'] = len(rail)

    # Bus stops for all areas (static data, no API call)
    result['bus_stops'] = {code: DUBLIN_BUS_STOPS.get(code, []) for code in area_codes}

    return jsonify(result)


@app.route('/api/prediction')
def api_prediction():
    """
    Predict delay risk for Bus & Luas based on current weather,
    using correlation coefficients from the historical DB.

    Query params: temp, rain, wind (current weather values)
    Returns delay risk level 0-4 for each mode + reasoning.
    """
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found. Run the pipeline first.'}), 404

    temp = float(request.args.get('temp', 10))
    rain = float(request.args.get('rain', 0))    # mm in last hour
    wind = float(request.args.get('wind', 10))   # km/h

    try:
        df = pd.read_sql_query('SELECT * FROM weather_transport_merged', conn)
        conn.close()
        if df.empty:
            return jsonify({'error': 'No data'}), 404

        # Build delay-proxy columns (same definition as /api/stats):
        # % shortfall below seasonal baseline ridership.
        for mode, col in [('bus', 'bus_passengers'), ('luas', 'luas_passengers')]:
            baseline = df.groupby('season')[col].transform('mean')
            df[f'{mode}_delay'] = ((baseline - df[col]) / baseline * 100).clip(lower=0)

        hourly_as_monthly = rain * 24 * 30  # rough mm/h → monthly-mm scale

        risks = {}
        for mode in ['bus', 'luas']:
            delay_col = f'{mode}_delay'
            reasoning = []
            risk_score = 0

            # Rainfall contribution (corr with delay proxy; positive → more delay)
            if df['total_rain'].std() > 0 and df[delay_col].std() > 0:
                r_rain = df['total_rain'].corr(df[delay_col])
                if rain > 0.5:
                    rain_pct = min(hourly_as_monthly / df['total_rain'].quantile(0.9), 2.0)
                    contribution = max(r_rain, 0) * rain_pct * 3.0
                    risk_score += contribution
                    reasoning.append({
                        'factor': 'Rainfall',
                        'current': f'{rain:.1f} mm/h',
                        'correlation': round(r_rain, 3),
                        'effect': 'increases delay' if r_rain > 0 else 'no delay effect',
                        'contribution': round(contribution, 2),
                    })

            # Wind contribution
            if df['avg_wind'].std() > 0:
                r_wind = df['avg_wind'].corr(df[delay_col])
                if wind > df['avg_wind'].quantile(0.5):
                    wind_pct = min(wind / df['avg_wind'].quantile(0.9), 2.0)
                    contribution = max(r_wind, 0) * wind_pct * 2.5
                    risk_score += contribution
                    reasoning.append({
                        'factor': 'Wind speed',
                        'current': f'{wind:.1f} km/h',
                        'correlation': round(r_wind, 3),
                        'effect': 'increases delay' if r_wind > 0 else 'no delay effect',
                        'contribution': round(contribution, 2),
                    })

            # Cold contribution (temp < 5°C adds risk)
            if temp < 5:
                cold_contribution = (5 - temp) / 5 * 2.0
                risk_score += cold_contribution
                reasoning.append({
                    'factor': 'Cold temperature',
                    'current': f'{temp:.1f}°C',
                    'correlation': None,
                    'effect': 'direct',
                    'contribution': round(cold_contribution, 2),
                })

            # Map to 0-4 risk level
            level = 0 if risk_score < 1 else \
                    1 if risk_score < 2.5 else \
                    2 if risk_score < 4 else \
                    3 if risk_score < 5.5 else 4
            labels = ['Normal', 'Slight delay possible', 'Moderate delay likely',
                      'High delay risk', 'Severe disruption likely']

            risks[mode] = {
                'level': level,
                'label': labels[level],
                'score': round(min(risk_score, 10), 2),
                'reasoning': reasoning,
            }

        # ── Area-level crowdedness prediction (weather vs passenger counts) ──
        severity = compute_weather_severity(temp, rain, wind)
        cscore = 0
        for var, cur_val, weight in [
            ('total_rain', hourly_as_monthly, 1.5),
            ('avg_wind', wind, 1.0),
            ('mean_temp', temp, 1.0),
            ('avg_severity', severity, 1.5),
        ]:
            if var in df.columns and df[var].std() > 0:
                r = df[var].corr(df['bus_passengers'])
                q90 = max(df[var].quantile(0.9), 1)
                pct = min(cur_val / q90, 2.0)
                cscore += r * pct * weight

        hour = datetime.now().hour
        is_rush = (7 <= hour <= 9) or (16 <= hour <= 19)
        is_late = hour < 6 or hour >= 23
        crowd_base = 3.0 + cscore
        if is_rush:
            crowd_base += 1.5
        if is_late:
            crowd_base -= 1.0
        crowd_level = max(1, min(5, round(crowd_base)))
        crowd_labels = {1: 'Empty', 2: 'Quiet', 3: 'Moderate', 4: 'Busy', 5: 'Very Busy'}

        return jsonify({
            'inputs': {'temp': temp, 'rain': rain, 'wind': wind},
            'bus': risks['bus'],
            'luas': risks['luas'],
            'crowdedness': {
                'level': crowd_level,
                'label': crowd_labels[crowd_level],
                'is_rush_hour': is_rush,
            },
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Bus stops by Dublin area — common routes & stops for each postal district
# Each route has direction info for realistic arrival boards
DUBLIN_BUS_STOPS = {
    'D1': [
        {'stop': "O'Connell St Upper", 'stop_id': '273',
         'routes': [
             {'route': '1', 'towards': 'Santry'},
             {'route': '11', 'towards': 'Wadelai Park'},
             {'route': '13', 'towards': 'Harristown'},
             {'route': '16', 'towards': 'Dublin Airport'},
             {'route': '38', 'towards': 'Damastown'},
             {'route': '39', 'towards': 'Ongar'},
             {'route': '46a', 'towards': 'Phoenix Park'},
         ]},
        {'stop': "O'Connell St Lower", 'stop_id': '270',
         'routes': [
             {'route': '1', 'towards': 'Shanard Road'},
             {'route': '16', 'towards': 'Ballinteer'},
             {'route': '41', 'towards': 'Swords'},
             {'route': '122', 'towards': 'Ashington'},
         ]},
        {'stop': 'Parnell Square', 'stop_id': '2',
         'routes': [
             {'route': '3', 'towards': 'Belfield'},
             {'route': '10', 'towards': 'Dun Laoghaire'},
             {'route': '40', 'towards': 'Liffey Valley'},
             {'route': '38a', 'towards': 'Damastown'},
         ]},
        {'stop': 'Connolly Station', 'stop_id': '90',
         'routes': [
             {'route': '1', 'towards': 'Santry'},
             {'route': '53', 'towards': 'Dublin Airport'},
             {'route': '130', 'towards': 'Clontarf'},
             {'route': '151', 'towards': 'Foxborough'},
         ]},
        {'stop': 'Busaras', 'stop_id': '317',
         'routes': [
             {'route': '27', 'towards': 'Clarehall'},
             {'route': '33', 'towards': 'Balbriggan'},
             {'route': '42', 'towards': 'Sandymount'},
             {'route': '151', 'towards': 'Docklands'},
         ]},
    ],
    'D3': [
        {'stop': 'Clontarf Road', 'stop_id': '576',
         'routes': [
             {'route': '130', 'towards': 'City Centre'},
             {'route': '14', 'towards': 'Beaumont'},
             {'route': '15', 'towards': 'Clongriffin'},
         ]},
        {'stop': 'Fairview', 'stop_id': '134',
         'routes': [
             {'route': '14', 'towards': 'Dundrum'},
             {'route': '27', 'towards': 'Clarehall'},
             {'route': '42', 'towards': 'Malahide'},
             {'route': '43', 'towards': 'Swords'},
         ]},
        {'stop': 'Howth Road (Clontarf)', 'stop_id': '578',
         'routes': [
             {'route': '31', 'towards': 'Howth Summit'},
             {'route': '32', 'towards': 'Malahide'},
             {'route': '130', 'towards': 'Lower Abbey St'},
         ]},
    ],
    'D5': [
        {'stop': 'Raheny Village', 'stop_id': '1076',
         'routes': [
             {'route': '29a', 'towards': 'City Centre'},
             {'route': '31', 'towards': 'Howth Summit'},
             {'route': '32', 'towards': 'Malahide'},
         ]},
        {'stop': 'Harmonstown Road', 'stop_id': '1074',
         'routes': [
             {'route': '6', 'towards': 'Maynooth'},
             {'route': '14', 'towards': 'Beaumont'},
         ]},
        {'stop': 'Watermill Road', 'stop_id': '1080',
         'routes': [
             {'route': '29a', 'towards': 'Lwr Abbey St'},
             {'route': '31', 'towards': 'Talbot St'},
         ]},
    ],
    'D7': [
        {'stop': 'Phibsborough Road', 'stop_id': '517',
         'routes': [
             {'route': '4', 'towards': 'Harristown'},
             {'route': '9', 'towards': 'Limekiln Ave'},
             {'route': '38', 'towards': 'Damastown'},
             {'route': '46a', 'towards': 'Dun Laoghaire'},
             {'route': '120', 'towards': 'Ashtown'},
         ]},
        {'stop': 'Cabra Road', 'stop_id': '1312',
         'routes': [
             {'route': '38', 'towards': 'City Centre'},
             {'route': '38a', 'towards': 'Damastown'},
             {'route': '39', 'towards': 'Ongar'},
         ]},
        {'stop': 'North Circular Road', 'stop_id': '520',
         'routes': [
             {'route': '4', 'towards': 'City Centre'},
             {'route': '46a', 'towards': 'Phoenix Park'},
             {'route': '120', 'towards': 'Parnell St'},
         ]},
    ],
    'D9': [
        {'stop': 'Drumcondra Road', 'stop_id': '109',
         'routes': [
             {'route': '1', 'towards': 'Santry'},
             {'route': '11', 'towards': 'City Centre'},
             {'route': '13', 'towards': 'Harristown'},
             {'route': '16', 'towards': 'Dublin Airport'},
             {'route': '33', 'towards': 'Swords'},
             {'route': '41', 'towards': 'Swords'},
         ]},
        {'stop': 'Griffith Avenue', 'stop_id': '529',
         'routes': [
             {'route': '9', 'towards': 'Limekiln Ave'},
             {'route': '13', 'towards': 'City Centre'},
             {'route': '17a', 'towards': 'Kilbarrack'},
         ]},
        {'stop': 'Botanic Road', 'stop_id': '526',
         'routes': [
             {'route': '4', 'towards': 'Harristown'},
             {'route': '9', 'towards': 'City Centre'},
         ]},
    ],
    'D11': [
        {'stop': 'Finglas Village', 'stop_id': '546',
         'routes': [
             {'route': '17a', 'towards': 'City Centre'},
             {'route': '40', 'towards': 'Liffey Valley'},
             {'route': '40b', 'towards': 'Tyrrelstown'},
             {'route': '40d', 'towards': 'Parnell St'},
         ]},
        {'stop': 'Clearwater SC', 'stop_id': '4627',
         'routes': [
             {'route': '17a', 'towards': 'Kilbarrack'},
             {'route': '40', 'towards': 'City Centre'},
             {'route': '140', 'towards': 'Rathmines'},
         ]},
        {'stop': 'Mellowes Road', 'stop_id': '549',
         'routes': [
             {'route': '40b', 'towards': 'Parnell St'},
             {'route': '40d', 'towards': 'Tyrrelstown'},
         ]},
    ],
    'D13': [
        {'stop': 'Donaghmede SC', 'stop_id': '2841',
         'routes': [
             {'route': '17', 'towards': 'City Centre'},
             {'route': '27', 'towards': 'Clarehall'},
             {'route': '42', 'towards': 'Malahide'},
             {'route': '43', 'towards': 'Swords'},
         ]},
        {'stop': 'Clarehall SC', 'stop_id': '2805',
         'routes': [
             {'route': '15', 'towards': 'Clongriffin'},
             {'route': '27', 'towards': 'City Centre'},
             {'route': '29a', 'towards': 'Lwr Abbey St'},
         ]},
        {'stop': 'Grange Road', 'stop_id': '2843',
         'routes': [
             {'route': '17', 'towards': 'Kilbarrack'},
             {'route': '27', 'towards': 'Busaras'},
         ]},
    ],
    'D15': [
        {'stop': 'Blanchardstown SC', 'stop_id': '4621',
         'routes': [
             {'route': '38', 'towards': 'City Centre'},
             {'route': '39', 'towards': 'Ongar'},
             {'route': '76a', 'towards': 'Talbot St'},
             {'route': '220', 'towards': 'Ballymun'},
         ]},
        {'stop': 'Castleknock Village', 'stop_id': '1569',
         'routes': [
             {'route': '37', 'towards': 'Blanchardstown'},
             {'route': '38', 'towards': 'Damastown'},
             {'route': '39', 'towards': 'City Centre'},
         ]},
        {'stop': 'Corduff Road', 'stop_id': '1583',
         'routes': [
             {'route': '38', 'towards': 'Damastown'},
             {'route': '76a', 'towards': 'City Centre'},
         ]},
    ],
    'D17': [
        {'stop': 'Coolock Village', 'stop_id': '2781',
         'routes': [
             {'route': '17', 'towards': 'City Centre'},
             {'route': '27', 'towards': 'Clarehall'},
             {'route': '42', 'towards': 'Malahide'},
         ]},
        {'stop': 'Northside SC', 'stop_id': '2771',
         'routes': [
             {'route': '27', 'towards': 'Busaras'},
             {'route': '42', 'towards': 'Sandymount'},
             {'route': '43', 'towards': 'Talbot St'},
         ]},
        {'stop': 'Oscar Traynor Road', 'stop_id': '2776',
         'routes': [
             {'route': '17', 'towards': 'Kilbarrack'},
             {'route': '42', 'towards': 'City Centre'},
         ]},
    ],
}

# Base frequencies (minutes between buses) by route, for rush vs off-peak
BUS_BASE_FREQ = {
    # High-frequency city routes
    '1': (6, 15), '11': (10, 20), '13': (8, 15), '14': (10, 20),
    '16': (8, 20), '27': (10, 20), '38': (6, 12), '39': (8, 15),
    '40': (8, 15), '41': (6, 12), '42': (10, 20), '43': (10, 20),
    '46a': (10, 20),
    # Medium-frequency
    '3': (12, 25), '4': (12, 20), '6': (15, 30), '9': (10, 20),
    '10': (12, 25), '15': (12, 25), '17': (12, 25), '17a': (12, 25),
    '29a': (12, 25), '31': (12, 20), '32': (15, 30), '33': (10, 20),
    '37': (15, 30), '38a': (12, 25), '53': (12, 25),
    # Lower-frequency
    '40b': (15, 30), '40d': (15, 30), '76a': (15, 30), '120': (15, 30),
    '122': (15, 30), '130': (10, 20), '140': (12, 25), '151': (15, 30),
    '220': (15, 30),
}


@app.route('/api/bus/stops')
def api_bus_stops():
    """Return bus stops and routes for selected Dublin area."""
    area = request.args.get('area', 'D1').upper()
    stops = DUBLIN_BUS_STOPS.get(area, DUBLIN_BUS_STOPS.get('D1', []))
    # Flatten for backward compatibility (frontend expects simple route list)
    simple = []
    for s in stops:
        simple.append({
            'stop': s['stop'],
            'stop_id': s.get('stop_id', ''),
            'routes': [r['route'] if isinstance(r, dict) else r for r in s['routes']],
        })
    return jsonify({'area': area, 'stops': simple})


@app.route('/api/bus/arrivals')
def api_bus_arrivals():
    """
    Estimated bus arrivals for a specific stop, with weather-based
    crowdedness and delay predictions.

    Uses scheduled frequencies adjusted by: time-of-day, weather severity,
    and historical delay-proxy correlations from the DB.

    Query params: area (D1..D17), stop_idx (0-based index in area's stop list)
    """
    import random

    area = request.args.get('area', 'D1').upper()
    stop_idx = int(request.args.get('stop_idx', 0))
    stops = DUBLIN_BUS_STOPS.get(area, [])
    if stop_idx < 0 or stop_idx >= len(stops):
        return jsonify({'error': 'Invalid stop index'}), 400

    stop_data = stops[stop_idx]
    now = datetime.now()
    hour = now.hour

    # Is it rush hour?
    is_rush = (7 <= hour <= 9) or (16 <= hour <= 19)
    is_late_night = hour < 6 or hour >= 23

    # Get current weather for this area (single API call for both severity + correlation scoring)
    cur_temp, cur_rain, cur_wind = 10, 0, 10
    weather_severity = 2.0
    try:
        import requests as _req
        resp = _req.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': str(DUBLIN_AREAS[area]['lat']),
            'longitude': str(DUBLIN_AREAS[area]['lon']),
            'current': 'temperature_2m,rain,wind_speed_10m',
            'timezone': 'Europe/Dublin'
        }, timeout=5)
        c = resp.json().get('current', {})
        cur_temp = c.get('temperature_2m', 10)
        cur_rain = c.get('rain', 0)
        cur_wind = c.get('wind_speed_10m', 10)
        weather_severity = compute_weather_severity(cur_temp, cur_rain, cur_wind)
    except Exception:
        pass

    # ── Correlation-based scores from historical data ──────────────
    # Both crowdedness and delay are driven by Pearson correlations
    # between weather variables and historical passenger / delay-proxy data.
    crowd_score = 2.0   # neutral baseline
    delay_score = 0.0   # minutes of weather-driven delay

    try:
        conn = get_db()
        if conn:
            cdf = pd.read_sql_query('SELECT * FROM weather_transport_merged', conn)
            conn.close()
            if not cdf.empty and 'bus_passengers' in cdf.columns:

                # Build delay-proxy column (same as /api/prediction)
                baseline = cdf.groupby('season')['bus_passengers'].transform('mean')
                cdf['bus_delay'] = ((baseline - cdf['bus_passengers']) / baseline * 100).clip(lower=0)

                rain_monthly = cur_rain * 24 * 30  # hourly mm → monthly scale

                # ── Crowdedness: weather vs passenger counts ──
                col = 'bus_passengers'
                cscore = 0
                for var, cur_val, weight in [
                    ('total_rain', rain_monthly, 1.5),
                    ('avg_wind', cur_wind, 1.0),
                    ('mean_temp', cur_temp, 1.0),
                    ('avg_severity', weather_severity, 1.5),
                ]:
                    if var in cdf.columns and cdf[var].std() > 0:
                        r = cdf[var].corr(cdf[col])
                        q90 = max(cdf[var].quantile(0.9), 1)
                        pct = min(cur_val / q90, 2.0)
                        cscore += r * pct * weight
                crowd_score = 3.0 + cscore

                # ── Delay: weather vs delay-proxy (% ridership shortfall) ──
                dcol = 'bus_delay'
                dscore = 0
                # Rainfall → delay correlation
                if cdf['total_rain'].std() > 0 and cdf[dcol].std() > 0:
                    r = cdf['total_rain'].corr(cdf[dcol])
                    pct = min(rain_monthly / max(cdf['total_rain'].quantile(0.9), 1), 2.0)
                    dscore += max(r, 0) * pct * 3.0
                # Wind → delay correlation
                if cdf['avg_wind'].std() > 0:
                    r = cdf['avg_wind'].corr(cdf[dcol])
                    pct = min(cur_wind / max(cdf['avg_wind'].quantile(0.9), 1), 2.0)
                    dscore += max(r, 0) * pct * 2.5
                # Cold temperature direct effect
                if cur_temp < 5:
                    dscore += (5 - cur_temp) / 5 * 2.0
                # Severity → delay correlation
                if 'avg_severity' in cdf.columns and cdf['avg_severity'].std() > 0:
                    r = cdf['avg_severity'].corr(cdf[dcol])
                    pct = min(weather_severity / max(cdf['avg_severity'].quantile(0.9), 1), 2.0)
                    dscore += max(r, 0) * pct * 2.0

                # Map delay_score to estimated extra minutes (0-8 range)
                # dscore is roughly 0..10, scale to realistic delay minutes
                delay_score = min(dscore * 0.8, 8.0)
    except Exception:
        pass

    # Rush hour adjustments
    if is_rush:
        crowd_score += 1.5
        delay_score += 1.5   # rush hour adds ~1.5 min baseline delay
    if is_late_night:
        crowd_score -= 1.0

    # Use deterministic seed so results are consistent within same minute
    seed = int(now.timestamp() // 60) + hash(stop_data['stop'])
    rng = random.Random(seed)

    arrivals = []
    for route_info in stop_data['routes']:
        route = route_info['route'] if isinstance(route_info, dict) else route_info
        towards = route_info.get('towards', 'City Centre') if isinstance(route_info, dict) else ''

        # Base frequency
        rush_freq, offpeak_freq = BUS_BASE_FREQ.get(route, (12, 25))
        base_freq = rush_freq if is_rush else offpeak_freq
        if is_late_night:
            base_freq = offpeak_freq * 2  # much less frequent at night

        # Weather delay factor: severity 0-10 maps to 0-40% extra delay on frequency
        weather_delay_pct = weather_severity * 4.0
        adjusted_freq = base_freq * (1.0 + weather_delay_pct / 100)

        # Generate next 3 arrivals for this route
        for i in range(3):
            # ETA = offset from now, with some randomness
            base_eta = adjusted_freq * (i * 0.8 + 0.3) + rng.uniform(-2, 3)
            eta_min = max(1, round(base_eta))

            # Delay: correlation-based score + small per-trip variation (±1 min)
            trip_variation = rng.uniform(-1.0, 1.0)
            base_delay = max(0, round(delay_score + trip_variation))

            # Crowdedness: 1-5 scale from correlation-based score
            crowd = max(1, min(5, round(crowd_score)))

            crowd_labels = {
                1: 'Empty', 2: 'Quiet', 3: 'Moderate', 4: 'Busy', 5: 'Very Busy'
            }

            arrivals.append({
                'route': route,
                'towards': towards,
                'eta_min': eta_min,
                'scheduled': eta_min - base_delay,
                'delay_min': max(0, base_delay),
                'crowdedness': crowd,
                'crowd_label': crowd_labels[crowd],
                'on_time': base_delay <= 1,
            })

    # Sort by ETA
    arrivals.sort(key=lambda a: a['eta_min'])

    return jsonify({
        'area': area,
        'stop': stop_data['stop'],
        'stop_id': stop_data.get('stop_id', ''),
        'weather_severity': round(weather_severity, 1),
        'is_rush_hour': is_rush,
        'arrivals': arrivals[:12],  # Max 12 arrivals
        'timestamp': now.isoformat(),
    })


@app.route('/api/tests/list')
def api_tests_list():
    """List all tests in the suite (class, method, docstring) without running them."""
    import unittest
    import tests as tests_module

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(tests_module)

    test_items = []
    def walk(s):
        for t in s:
            if isinstance(t, unittest.TestSuite):
                walk(t)
            else:
                cls = type(t).__name__
                method = t._testMethodName
                doc = (getattr(t, method).__doc__ or '').strip().split('\n')[0]
                test_items.append({
                    'class': cls,
                    'method': method,
                    'description': doc
                })

    walk(suite)
    # Group by class
    by_class = {}
    for t in test_items:
        by_class.setdefault(t['class'], []).append(
            {'method': t['method'], 'description': t['description']})
    return jsonify({
        'total': len(test_items),
        'classes': [{'name': k, 'tests': v} for k, v in sorted(by_class.items())]
    })


@app.route('/api/dashboard/data')
def api_dashboard():
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found'}), 404

    try:
        merged = pd.read_sql_query(
            'SELECT * FROM weather_transport_merged ORDER BY year, month', conn)
        seasonal = pd.read_sql_query("""
            SELECT season, AVG(bus_passengers) as avg_bus, AVG(luas_passengers) as avg_luas,
                   AVG(total_rain) as avg_rain, AVG(mean_temp) as avg_temp,
                   AVG(avg_severity) as avg_severity
            FROM weather_transport_merged GROUP BY season
        """, conn)
        severity = pd.read_sql_query("""
            SELECT rain_category, COUNT(*) as days,
                   AVG(avg_severity) as avg_severity, AVG(avg_temp) as avg_temp,
                   AVG(total_rain) as avg_rain
            FROM weather_daily GROUP BY rain_category
        """, conn)
        bus = pd.read_sql_query(
            'SELECT * FROM bus_passengers ORDER BY year, month', conn)
        luas = pd.read_sql_query(
            'SELECT * FROM luas_passengers ORDER BY year, month', conn)
        # Weather severity over time (monthly)
        weather_trend = pd.read_sql_query("""
            SELECT year, month, mean_temp, total_rain, avg_severity, rainy_days, severe_days
            FROM weather_monthly WHERE year >= 2018 ORDER BY year, month
        """, conn)
        # Impact analysis: ridership by weather severity group
        impact = pd.read_sql_query("""
            SELECT severity_group,
                   AVG(bus_passengers) as avg_bus, AVG(luas_passengers) as avg_luas,
                   AVG(avg_severity) as avg_severity, COUNT(*) as months
            FROM weather_transport_merged GROUP BY severity_group
        """, conn)

        conn.close()

        def rec(df):
            return json.loads(df.to_json(orient='records'))

        return jsonify({
            'merged': rec(merged), 'seasonal': rec(seasonal),
            'severity': rec(severity), 'bus_timeline': rec(bus),
            'luas_timeline': rec(luas), 'weather_trend': rec(weather_trend),
            'impact': rec(impact)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/features/sample')
def api_features_sample():
    """Return a sample of each feature extraction table for the Feature Extraction showcase."""
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found. Run the pipeline first.'}), 404

    try:
        result = {}
        # Hourly with temporal + weather features (latest 24 records)
        hourly = pd.read_sql_query("""
            SELECT timestamp, temp_c, rain_mm, wind_speed_kmh,
                   season, is_weekend, is_rush_hour,
                   rain_category, weather_severity, comfort_index, temp_category
            FROM weather_hourly
            ORDER BY timestamp DESC LIMIT 24
        """, conn)

        # Daily aggregates (latest 14)
        daily = pd.read_sql_query("""
            SELECT date, avg_temp, total_rain, avg_wind, rain_hours,
                   rain_category, max_severity, rain_intensity
            FROM weather_daily ORDER BY date DESC LIMIT 14
        """, conn)

        # Monthly aggregates (latest 12)
        monthly = pd.read_sql_query("""
            SELECT year, month, mean_temp, total_rain, rainy_days,
                   severe_days, avg_severity, season
            FROM weather_monthly ORDER BY year DESC, month DESC LIMIT 12
        """, conn)

        # Transport features (latest 12 per mode)
        bus = pd.read_sql_query("""
            SELECT year, month, passengers, passengers_norm,
                   yoy_change, rolling_avg_3m, season, is_covid
            FROM bus_passengers ORDER BY year DESC, month DESC LIMIT 12
        """, conn)
        luas = pd.read_sql_query("""
            SELECT year, month, passengers, passengers_norm,
                   yoy_change, rolling_avg_3m, season, is_covid
            FROM luas_passengers ORDER BY year DESC, month DESC LIMIT 12
        """, conn)

        # Merged interaction features
        merged = pd.read_sql_query("""
            SELECT year, month, season, bus_passengers, luas_passengers,
                   total_passengers, bus_share, total_rain, mean_temp,
                   avg_severity, rain_group, temp_group, severity_group,
                   weather_impact_score
            FROM weather_transport_merged ORDER BY year DESC, month DESC LIMIT 12
        """, conn)

        conn.close()

        def rec(df):
            return {'columns': list(df.columns),
                    'rows': json.loads(df.to_json(orient='records'))}

        return jsonify({
            'hourly': rec(hourly), 'daily': rec(daily),
            'monthly': rec(monthly), 'bus': rec(bus),
            'luas': rec(luas), 'merged': rec(merged)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Weather Impact on Dublin Transport — Pipeline Server")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
