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

        results = {}
        for mode, col in [('bus', 'bus_passengers'), ('luas', 'luas_passengers')]:
            correlations = []
            for var, label in [('total_rain', 'Rainfall (mm)'), ('mean_temp', 'Temperature (°C)'),
                               ('avg_wind', 'Wind Speed (km/h)'), ('rainy_days', 'Rainy Days'),
                               ('avg_severity', 'Weather Severity')]:
                if var in df.columns and col in df.columns:
                    valid = df[[var, col]].dropna()
                    if len(valid) > 5:
                        pr, pp = pearsonr(valid[var], valid[col])
                        sr, sp = spearmanr(valid[var], valid[col])
                        correlations.append({
                            'variable': label, 'pearson_r': round(pr, 4),
                            'pearson_p': round(pp, 4), 'spearman_r': round(sr, 4),
                            'spearman_p': round(sp, 4), 'significant': pp < 0.05
                        })
            results[mode] = correlations

        # Kruskal-Wallis
        kw_tests = []
        for mode, col in [('Bus', 'bus_passengers'), ('Luas', 'luas_passengers')]:
            for gcol, glabel in [('rain_group', 'Rainfall'), ('temp_group', 'Temperature'),
                                  ('severity_group', 'Severity')]:
                if gcol in df.columns:
                    groups = [g[col].dropna().values for _, g in df.groupby(gcol)]
                    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                        stat, p = kruskal(*groups)
                        kw_tests.append({
                            'test': f'{mode} × {glabel}',
                            'statistic': round(stat, 3),
                            'p_value': round(p, 4),
                            'significant': p < 0.05
                        })
        results['kruskal_wallis'] = kw_tests

        results['summary'] = {
            'total_records': len(df),
            'year_range': f"{int(df['year'].min())}–{int(df['year'].max())}",
            'avg_bus': int(df['bus_passengers'].mean()) if 'bus_passengers' in df.columns else 0,
            'avg_luas': int(df['luas_passengers'].mean()) if 'luas_passengers' in df.columns else 0,
            'avg_rain': round(df['total_rain'].mean(), 1),
            'avg_temp': round(df['mean_temp'].mean(), 1),
            'avg_severity': round(df['avg_severity'].mean(), 2) if 'avg_severity' in df.columns else 0
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


@app.route('/api/live')
def api_live():
    """
    Live data endpoint — called every 60s by the dashboard.
    Returns current weather (Open-Meteo free, no key), Luas real-time, Irish Rail real-time.
    """
    result = {'timestamp': datetime.now().isoformat()}

    # Current weather from Open-Meteo (free, no API key needed)
    try:
        import requests as _req
        resp = _req.get('https://api.open-meteo.com/v1/forecast', params={
            'latitude': 53.3498, 'longitude': -6.2603,
            'current': 'temperature_2m,relative_humidity_2m,rain,wind_speed_10m,'
                       'surface_pressure,cloud_cover,weather_code',
            'timezone': 'Europe/Dublin'
        }, timeout=8)
        resp.raise_for_status()
        c = resp.json().get('current', {})
        # Map WMO weather codes to descriptions
        wmo = {0: 'Clear sky', 1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
               45: 'Foggy', 48: 'Depositing rime fog', 51: 'Light drizzle',
               53: 'Moderate drizzle', 55: 'Dense drizzle', 61: 'Slight rain',
               63: 'Moderate rain', 65: 'Heavy rain', 71: 'Slight snow',
               73: 'Moderate snow', 75: 'Heavy snow', 80: 'Slight showers',
               81: 'Moderate showers', 82: 'Violent showers', 95: 'Thunderstorm'}
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
            'description': wmo.get(code, f'Code {code}'),
            'weather_code': code,
            'severity': severity,
        }
    except Exception as e:
        result['weather'] = {'error': str(e)}

    # Luas real-time (key stops for speed)
    try:
        luas = fetch_luas_all_stops(key_only=True)
        result['luas'] = luas
        result['luas_count'] = len(luas)
    except Exception as e:
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
    except Exception as e:
        result['rail'] = []
        result['rail_count'] = 0

    return jsonify(result)


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


@app.route('/api/dashboard/simulate')
def api_simulate():
    """Weather impact simulator — predict ridership impact for given conditions."""
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found'}), 404

    temp = float(request.args.get('temp', 10))
    rain = float(request.args.get('rain', 50))
    wind = float(request.args.get('wind', 20))

    try:
        df = pd.read_sql_query('SELECT * FROM weather_transport_merged', conn)
        conn.close()

        if df.empty or 'mean_temp' not in df.columns:
            return jsonify({'error': 'No data available. Run the pipeline first.'}), 404

        # Find similar weather months
        df['distance'] = np.sqrt(
            ((df['mean_temp'] - temp) / 10) ** 2 +
            ((df['total_rain'] - rain) / 100) ** 2 +
            ((df['avg_wind'] - wind) / 30) ** 2
        )
        similar = df.nsmallest(5, 'distance')
        overall_avg_bus = df['bus_passengers'].mean()
        overall_avg_luas = df['luas_passengers'].mean()

        predicted_bus = int(similar['bus_passengers'].mean())
        predicted_luas = int(similar['luas_passengers'].mean())
        bus_impact = round((predicted_bus - overall_avg_bus) / overall_avg_bus * 100, 1)
        luas_impact = round((predicted_luas - overall_avg_luas) / overall_avg_luas * 100, 1)

        severity = compute_weather_severity(temp, rain / 30, wind)  # daily approx

        return jsonify({
            'predicted_bus': predicted_bus,
            'predicted_luas': predicted_luas,
            'bus_impact_pct': bus_impact,
            'luas_impact_pct': luas_impact,
            'severity': severity,
            'similar_months': json.loads(similar[['year', 'month', 'season',
                                                   'bus_passengers', 'luas_passengers',
                                                   'mean_temp', 'total_rain']].to_json(orient='records'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Weather Impact on Dublin Transport — Pipeline Server")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
