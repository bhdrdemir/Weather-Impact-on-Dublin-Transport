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
    'D2':  {'name': 'Dublin 2 — City South', 'lat': 53.3382, 'lon': -6.2522},
    'D4':  {'name': 'Dublin 4 — Ballsbridge', 'lat': 53.3266, 'lon': -6.2317},
    'D6':  {'name': 'Dublin 6 — Rathmines', 'lat': 53.3219, 'lon': -6.2645},
    'D7':  {'name': 'Dublin 7 — Phibsborough', 'lat': 53.3575, 'lon': -6.2729},
    'D8':  {'name': 'Dublin 8 — Kilmainham', 'lat': 53.3379, 'lon': -6.2957},
    'D9':  {'name': 'Dublin 9 — Drumcondra', 'lat': 53.3760, 'lon': -6.2552},
    'D14': {'name': 'Dublin 14 — Dundrum', 'lat': 53.2920, 'lon': -6.2419},
    'D15': {'name': 'Dublin 15 — Blanchardstown', 'lat': 53.3872, 'lon': -6.3782},
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
    Live data endpoint — called every 60s by the dashboard.
    Query parameter: area (default D1).
    Returns current weather + hourly forecast for the selected area,
    plus Luas real-time + Irish Rail real-time (same for all areas).
    """
    area_code = request.args.get('area', 'D1').upper()
    area = DUBLIN_AREAS.get(area_code, DUBLIN_AREAS['D1'])
    result = {
        'timestamp': datetime.now().isoformat(),
        'area': {'code': area_code, **area},
    }

    # Current weather + 24h hourly forecast (single Open-Meteo call)
    try:
        import requests as _req
        resp = _req.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': area['lat'], 'longitude': area['lon'],
            'current': 'temperature_2m,relative_humidity_2m,rain,wind_speed_10m,'
                       'surface_pressure,cloud_cover,weather_code',
            'hourly': 'temperature_2m,rain,wind_speed_10m,weather_code',
            'forecast_hours': 24,
            'timezone': 'Europe/Dublin'
        }, timeout=8)
        resp.raise_for_status()
        data = resp.json()
        c = data.get('current', {})
        code = c.get('weather_code', 0)
        severity = compute_weather_severity(
            c.get('temperature_2m', 10), c.get('rain', 0), c.get('wind_speed_10m', 10))
        result['weather'] = {
            'temp': c.get('temperature_2m'),
            'humidity': c.get('relative_humidity_2m'),
            'rain': c.get('rain', 0),
            'wind_speed': c.get('wind_speed_10m'),
            'pressure': c.get('surface_pressure'),
            'cloud_cover': c.get('cloud_cover'),
            'description': WMO_CODES.get(code, f'Code {code}'),
            'weather_code': code,
            'severity': severity,
            'observed_at': c.get('time'),
        }

        # Hourly forecast (next 24h)
        h = data.get('hourly', {})
        times = h.get('time', [])
        temps = h.get('temperature_2m', [])
        rains = h.get('rain', [])
        winds = h.get('wind_speed_10m', [])
        codes = h.get('weather_code', [])
        forecast = []
        for i in range(min(24, len(times))):
            forecast.append({
                'time': times[i],
                'temp': temps[i] if i < len(temps) else None,
                'rain': rains[i] if i < len(rains) else 0,
                'wind_speed': winds[i] if i < len(winds) else None,
                'weather_code': codes[i] if i < len(codes) else 0,
                'description': WMO_CODES.get(codes[i] if i < len(codes) else 0, ''),
            })
        result['forecast'] = forecast
    except Exception as e:
        result['weather'] = {'error': str(e)}
        result['forecast'] = []

    # Luas real-time (key stops for speed)
    try:
        luas = fetch_luas_all_stops(key_only=True)
        result['luas'] = luas
        result['luas_count'] = len(luas)
    except Exception:
        result['luas'] = []
        result['luas_count'] = 0

    # Irish Rail real-time
    try:
        rail_df = fetch_irish_rail_realtime()
        if not rail_df.empty:
            result['rail'] = json.loads(rail_df.to_json(orient='records'))
            result['rail_count'] = len(rail_df)
        else:
            result['rail'] = []
            result['rail_count'] = 0
    except Exception:
        result['rail'] = []
        result['rail_count'] = 0

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

        return jsonify({
            'inputs': {'temp': temp, 'rain': rain, 'wind': wind},
            'bus': risks['bus'],
            'luas': risks['luas'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


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
