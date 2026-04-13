"""
app.py — Flask Backend for Weather Impact on Dublin Transport Pipeline
======================================================================
B9AI001 Programming for Data Analytics — CA2

Routes:
  GET  /                    → Frontend (single-page app)
  POST /api/pipeline/run    → Execute full ETL pipeline
  GET  /api/pipeline/status → Pipeline status
  GET  /api/stats           → Correlation & summary statistics
  GET  /api/query           → Execute SQL query against the database
  GET  /api/tables          → List all database tables with row counts
  GET  /api/tests/run       → Run unit + integration tests
  GET  /api/weather/current → Fetch current weather from OpenWeatherMap
  GET  /api/dashboard/data  → Dashboard chart data
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
    run_full_pipeline, DB_PATH, fetch_openweather_current,
    classify_rainfall, map_season, compute_weather_severity
)
from tests import run_tests

app = Flask(__name__)

# Pipeline state (shared across requests)
pipeline_state = {
    'status': 'idle',
    'last_run': None,
    'result': None
}


def get_db():
    """Get a database connection (read-only for queries)."""
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
def run_pipeline():
    """Execute the full ETL pipeline (async)."""
    if pipeline_state['status'] == 'running':
        return jsonify({'error': 'Pipeline already running'}), 409

    api_key = request.json.get('api_key', '') if request.is_json else ''

    def _run():
        pipeline_state['status'] = 'running'
        pipeline_state['result'] = None
        try:
            result = run_full_pipeline(owm_api_key=api_key or None)
            pipeline_state['result'] = result
            pipeline_state['status'] = result.get('status', 'complete')
        except Exception as e:
            pipeline_state['status'] = 'failed'
            pipeline_state['result'] = {'error': str(e)}
        pipeline_state['last_run'] = datetime.now().isoformat()

    thread = threading.Thread(target=_run)
    thread.start()

    return jsonify({'message': 'Pipeline started', 'status': 'running'})


@app.route('/api/pipeline/status')
def pipeline_status():
    """Get current pipeline state."""
    return jsonify(pipeline_state)


@app.route('/api/stats')
def get_stats():
    """Compute correlation statistics from the database."""
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found. Run the pipeline first.'}), 404

    try:
        df = pd.read_sql_query('SELECT * FROM weather_transport_merged', conn)
        conn.close()

        if df.empty:
            return jsonify({'error': 'No data in merged table'}), 404

        results = {}
        for mode, col in [('bus', 'bus_passengers'), ('luas', 'luas_passengers')]:
            correlations = []
            weather_vars = {
                'rain': 'Rainfall (mm)',
                'meant': 'Mean Temperature (°C)',
                'wdsp': 'Wind Speed (kt)',
                'sun': 'Sunshine Hours'
            }
            for var, label in weather_vars.items():
                if var in df.columns and col in df.columns:
                    valid = df[[var, col]].dropna()
                    if len(valid) > 10:
                        pr, pp = pearsonr(valid[var], valid[col])
                        sr, sp = spearmanr(valid[var], valid[col])
                        correlations.append({
                            'variable': label,
                            'pearson_r': round(pr, 4),
                            'pearson_p': round(pp, 4),
                            'spearman_r': round(sr, 4),
                            'spearman_p': round(sp, 4),
                            'significant': pp < 0.05
                        })
            results[mode] = correlations

        # Kruskal-Wallis tests
        kw_tests = []
        for mode, col in [('bus', 'bus_passengers'), ('luas', 'luas_passengers')]:
            for group_col, group_label in [('rain_group', 'Rainfall'), ('temp_group', 'Temperature')]:
                if group_col in df.columns:
                    groups = [g[col].dropna().values for _, g in df.groupby(group_col)]
                    if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                        stat, p = kruskal(*groups)
                        kw_tests.append({
                            'test': f'{mode.title()} × {group_label}',
                            'statistic': round(stat, 3),
                            'p_value': round(p, 4),
                            'significant': p < 0.05
                        })
        results['kruskal_wallis'] = kw_tests

        # Summary stats
        results['summary'] = {
            'total_records': len(df),
            'year_range': f"{int(df['year'].min())}–{int(df['year'].max())}",
            'avg_bus': int(df['bus_passengers'].mean()) if 'bus_passengers' in df.columns else 0,
            'avg_luas': int(df['luas_passengers'].mean()) if 'luas_passengers' in df.columns else 0,
            'avg_rain': round(df['rain'].mean(), 1),
            'avg_temp': round(df['meant'].mean(), 1)
        }

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/query')
def execute_query():
    """Execute a read-only SQL query."""
    sql = request.args.get('sql', '').strip()
    if not sql:
        return jsonify({'error': 'No SQL query provided'}), 400

    # Safety: only allow SELECT queries
    if not sql.upper().startswith('SELECT'):
        return jsonify({'error': 'Only SELECT queries are allowed'}), 403

    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found. Run the pipeline first.'}), 404

    try:
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return jsonify({
            'columns': list(df.columns),
            'data': df.head(500).to_dict(orient='records'),
            'total_rows': len(df)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/tables')
def list_tables():
    """List all tables with row counts."""
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found'}), 404

    try:
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
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/tests/run')
def run_test_suite():
    """Run all unit and integration tests."""
    try:
        results = run_tests()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/weather/current')
def current_weather():
    """Fetch current Dublin weather from OpenWeatherMap."""
    api_key = request.args.get('api_key', '')
    if not api_key:
        return jsonify({'error': 'API key required'}), 400

    result = fetch_openweather_current(api_key)
    if result:
        return jsonify(result)
    return jsonify({'error': 'Failed to fetch weather'}), 502


@app.route('/api/dashboard/data')
def dashboard_data():
    """Get pre-processed data for dashboard charts."""
    conn = get_db()
    if not conn:
        return jsonify({'error': 'Database not found'}), 404

    try:
        # Monthly ridership trends
        merged = pd.read_sql_query(
            'SELECT * FROM weather_transport_merged ORDER BY year, month', conn)

        # Seasonal averages
        seasonal = pd.read_sql_query("""
            SELECT season,
                   AVG(bus_passengers) as avg_bus,
                   AVG(luas_passengers) as avg_luas,
                   AVG(rain) as avg_rain,
                   AVG(meant) as avg_temp
            FROM weather_transport_merged
            GROUP BY season
        """, conn)

        # Weather severity distribution (recent years)
        severity = pd.read_sql_query("""
            SELECT rain_category, COUNT(*) as days,
                   AVG(weather_severity) as avg_severity,
                   AVG(avg_temp) as avg_temp
            FROM weather_daily
            WHERE year >= 2018
            GROUP BY rain_category
        """, conn)

        # Bus timeline
        bus_timeline = pd.read_sql_query(
            'SELECT year, month, passengers, season FROM bus_passengers ORDER BY year, month',
            conn)

        # Luas timeline
        luas_timeline = pd.read_sql_query(
            'SELECT year, month, passengers, season FROM luas_passengers ORDER BY year, month',
            conn)

        conn.close()

        def safe_records(df):
            return json.loads(df.to_json(orient='records'))

        return jsonify({
            'merged': safe_records(merged),
            'seasonal': safe_records(seasonal),
            'severity': safe_records(severity),
            'bus_timeline': safe_records(bus_timeline),
            'luas_timeline': safe_records(luas_timeline)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ===================================================================
# Run the server
# ===================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("  Weather Impact on Dublin Transport — Pipeline Server")
    print("  Open http://127.0.0.1:5000 in your browser")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
