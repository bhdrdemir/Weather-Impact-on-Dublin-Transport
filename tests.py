"""
tests.py — Unit Tests + Integration Test
=========================================
Weather Impact on Dublin Public Transport — ETL Pipeline
B9AI001 Programming for Data Analytics — CA2

Tests cover all pipeline helper functions, feature extraction,
and a full integration round-trip (create → transform → store → query).
"""

import unittest
import numpy as np
import pandas as pd
import sqlite3
import os
import tempfile
import json

from pipeline import (
    classify_rainfall, map_season, compute_weather_severity,
    normalise_passengers, validate_dataframe,
    extract_temporal_features, extract_weather_features,
    extract_daily_aggregates, extract_monthly_aggregates,
    extract_merged_features, extract_transport_features,
    create_database, load_to_database,
    DB_PATH, SEASON_MAP
)


class TestClassifyRainfall(unittest.TestCase):
    """Unit tests for rainfall classification function."""

    def test_dry(self):
        self.assertEqual(classify_rainfall(0.0), 'dry')
        self.assertEqual(classify_rainfall(0.1), 'dry')
        self.assertEqual(classify_rainfall(0.2), 'dry')

    def test_light(self):
        self.assertEqual(classify_rainfall(0.3), 'light')
        self.assertEqual(classify_rainfall(2.5), 'light')
        self.assertEqual(classify_rainfall(5.0), 'light')

    def test_moderate(self):
        self.assertEqual(classify_rainfall(5.1), 'moderate')
        self.assertEqual(classify_rainfall(10.0), 'moderate')
        self.assertEqual(classify_rainfall(15.0), 'moderate')

    def test_heavy(self):
        self.assertEqual(classify_rainfall(15.1), 'heavy')
        self.assertEqual(classify_rainfall(30.0), 'heavy')

    def test_very_heavy(self):
        self.assertEqual(classify_rainfall(30.1), 'very_heavy')
        self.assertEqual(classify_rainfall(100.0), 'very_heavy')

    def test_nan_returns_unknown(self):
        self.assertEqual(classify_rainfall(np.nan), 'unknown')

    def test_negative_returns_unknown(self):
        self.assertEqual(classify_rainfall(-1.0), 'unknown')


class TestMapSeason(unittest.TestCase):
    """Unit tests for season mapping function."""

    def test_winter_months(self):
        for m in [12, 1, 2]:
            self.assertEqual(map_season(m), 'Winter')

    def test_spring_months(self):
        for m in [3, 4, 5]:
            self.assertEqual(map_season(m), 'Spring')

    def test_summer_months(self):
        for m in [6, 7, 8]:
            self.assertEqual(map_season(m), 'Summer')

    def test_autumn_months(self):
        for m in [9, 10, 11]:
            self.assertEqual(map_season(m), 'Autumn')

    def test_invalid_month(self):
        self.assertEqual(map_season(13), 'unknown')
        self.assertEqual(map_season(0), 'unknown')


class TestWeatherSeverity(unittest.TestCase):
    """Unit tests for weather severity index computation."""

    def test_good_weather_low_severity(self):
        severity = compute_weather_severity(15.0, 0.0, 5.0)
        self.assertLess(severity, 1.5)

    def test_bad_weather_high_severity(self):
        severity = compute_weather_severity(-5.0, 40.0, 70.0)
        self.assertGreater(severity, 8.0)

    def test_nan_input_returns_nan(self):
        self.assertTrue(np.isnan(compute_weather_severity(np.nan, 5, 10)))
        self.assertTrue(np.isnan(compute_weather_severity(10, np.nan, 10)))
        self.assertTrue(np.isnan(compute_weather_severity(10, 5, np.nan)))

    def test_output_within_range(self):
        for _ in range(100):
            t = np.random.uniform(-10, 30)
            r = np.random.uniform(0, 50)
            w = np.random.uniform(0, 80)
            s = compute_weather_severity(t, r, w)
            self.assertGreaterEqual(s, 0)
            self.assertLessEqual(s, 10)

    def test_moderate_conditions(self):
        severity = compute_weather_severity(10.0, 10.0, 20.0)
        self.assertGreater(severity, 1.0)
        self.assertLess(severity, 7.0)


class TestNormalisePassengers(unittest.TestCase):
    """Unit tests for passenger normalisation."""

    def test_minmax_bounds(self):
        s = pd.Series([100, 200, 300, 400, 500])
        result = normalise_passengers(s, 'minmax')
        self.assertAlmostEqual(result.iloc[0], 0.0)
        self.assertAlmostEqual(result.iloc[-1], 1.0)

    def test_minmax_midpoint(self):
        s = pd.Series([0, 50, 100])
        result = normalise_passengers(s, 'minmax')
        self.assertAlmostEqual(result.iloc[1], 0.5)

    def test_zscore_mean_zero(self):
        s = pd.Series([100, 200, 300])
        result = normalise_passengers(s, 'zscore')
        self.assertAlmostEqual(result.mean(), 0.0, places=5)

    def test_zscore_std_one(self):
        s = pd.Series([10, 20, 30, 40, 50])
        result = normalise_passengers(s, 'zscore')
        self.assertAlmostEqual(result.std(), 1.0, places=3)

    def test_invalid_method_raises(self):
        with self.assertRaises(ValueError):
            normalise_passengers(pd.Series([1, 2, 3]), 'invalid')

    def test_constant_series_minmax(self):
        s = pd.Series([5, 5, 5])
        result = normalise_passengers(s, 'minmax')
        self.assertTrue((result == 0.0).all())


class TestValidateDataframe(unittest.TestCase):
    """Unit tests for DataFrame validation."""

    def test_valid_df_passes(self):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = validate_dataframe(df, ['a', 'b'], 'test')
        self.assertTrue(result['passed'])
        self.assertEqual(result['rows'], 2)

    def test_missing_column_fails(self):
        df = pd.DataFrame({'a': [1, 2]})
        result = validate_dataframe(df, ['a', 'b'], 'test')
        self.assertFalse(result['passed'])
        self.assertIn('b', result['missing_cols'])

    def test_empty_df_fails(self):
        df = pd.DataFrame({'a': []})
        result = validate_dataframe(df, ['a'], 'test')
        self.assertFalse(result['passed'])

    def test_null_count_reported(self):
        df = pd.DataFrame({'x': [1, None, 3]})
        result = validate_dataframe(df, ['x'], 'test')
        self.assertEqual(result['null_counts']['x'], 1)


class TestExtractTemporalFeatures(unittest.TestCase):
    """Unit tests for temporal feature extraction (API-driven pipeline)."""

    def test_features_added(self):
        df = pd.DataFrame({'timestamp': pd.date_range('2023-06-15', periods=3, freq='h')})
        result = extract_temporal_features(df, 'timestamp')
        expected_cols = ['year', 'month', 'day', 'hour', 'weekday', 'day_of_week',
                         'is_weekend', 'season', 'quarter', 'month_name', 'is_rush_hour']
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_season_mapping(self):
        df = pd.DataFrame({'timestamp': [pd.Timestamp('2023-01-15'),
                                          pd.Timestamp('2023-07-15')]})
        result = extract_temporal_features(df, 'timestamp')
        self.assertEqual(result.iloc[0]['season'], 'Winter')
        self.assertEqual(result.iloc[1]['season'], 'Summer')

    def test_weekend_detection(self):
        df = pd.DataFrame({'timestamp': [pd.Timestamp('2023-06-17'),   # Saturday
                                          pd.Timestamp('2023-06-19')]}) # Monday
        result = extract_temporal_features(df, 'timestamp')
        self.assertTrue(result.iloc[0]['is_weekend'])
        self.assertFalse(result.iloc[1]['is_weekend'])

    def test_rush_hour_detection(self):
        """Verify rush hour: weekday 8AM = rush, weekday 12PM = not rush."""
        df = pd.DataFrame({'timestamp': [
            pd.Timestamp('2023-06-19 08:00'),  # Monday 8 AM — rush
            pd.Timestamp('2023-06-19 12:00'),  # Monday 12 PM — not rush
            pd.Timestamp('2023-06-17 08:00'),  # Saturday 8 AM — weekend, not rush
        ]})
        result = extract_temporal_features(df, 'timestamp')
        self.assertTrue(result.iloc[0]['is_rush_hour'])
        self.assertFalse(result.iloc[1]['is_rush_hour'])
        self.assertFalse(result.iloc[2]['is_rush_hour'])


class TestExtractWeatherFeatures(unittest.TestCase):
    """Unit tests for weather feature extraction (API-driven column names)."""

    def test_features_added(self):
        df = pd.DataFrame({
            'temp_c': [15.0, 20.0],
            'rain_mm': [0.0, 10.0],
            'wind_speed_kmh': [10.0, 20.0],
            'humidity': [70.0, 90.0],
        })
        result = extract_weather_features(df)
        self.assertIn('rain_category', result.columns)
        self.assertIn('is_rainy', result.columns)
        self.assertIn('wind_chill', result.columns)
        self.assertIn('weather_severity', result.columns)
        self.assertIn('comfort_index', result.columns)
        self.assertIn('temp_category', result.columns)
        self.assertIn('poor_visibility', result.columns)

    def test_rain_category_classification(self):
        df = pd.DataFrame({
            'temp_c': [10.0], 'rain_mm': [0.0],
            'wind_speed_kmh': [5.0], 'humidity': [60.0],
        })
        result = extract_weather_features(df)
        self.assertEqual(result.iloc[0]['rain_category'], 'dry')

    def test_comfort_index_complement(self):
        """comfort_index should be 10 - weather_severity."""
        df = pd.DataFrame({
            'temp_c': [12.0], 'rain_mm': [3.0],
            'wind_speed_kmh': [15.0], 'humidity': [75.0],
        })
        result = extract_weather_features(df)
        self.assertAlmostEqual(
            result.iloc[0]['comfort_index'],
            10 - result.iloc[0]['weather_severity'],
            places=2
        )

    def test_poor_visibility_flag(self):
        """High humidity + rain → poor visibility."""
        df = pd.DataFrame({
            'temp_c': [8.0], 'rain_mm': [5.0],
            'wind_speed_kmh': [10.0], 'humidity': [95.0],
        })
        result = extract_weather_features(df)
        self.assertTrue(result.iloc[0]['poor_visibility'])


class TestExtractDailyAggregates(unittest.TestCase):
    """Unit tests for daily aggregation of hourly weather data."""

    def setUp(self):
        """Create a small hourly DataFrame spanning 2 days."""
        np.random.seed(42)
        timestamps = pd.date_range('2023-06-01', periods=48, freq='h')
        self.hourly = pd.DataFrame({
            'timestamp': timestamps,
            'temp_c': np.random.uniform(10, 20, 48),
            'rain_mm': np.random.exponential(1.0, 48),
            'wind_speed_kmh': np.random.uniform(5, 25, 48),
            'humidity': np.random.uniform(60, 95, 48),
            'pressure_hpa': np.random.uniform(1000, 1020, 48),
            'cloud_cover': np.random.uniform(0, 100, 48),
        })
        self.hourly = extract_weather_features(self.hourly)

    def test_output_has_two_days(self):
        daily = extract_daily_aggregates(self.hourly)
        self.assertEqual(len(daily), 2)

    def test_daily_columns_present(self):
        daily = extract_daily_aggregates(self.hourly)
        for col in ['avg_temp', 'total_rain', 'avg_wind', 'rain_hours',
                     'temp_range', 'rain_category', 'rain_intensity']:
            self.assertIn(col, daily.columns, f"Missing column: {col}")

    def test_total_rain_positive(self):
        daily = extract_daily_aggregates(self.hourly)
        self.assertTrue((daily['total_rain'] >= 0).all())


class TestExtractMonthlyAggregates(unittest.TestCase):
    """Unit tests for monthly aggregation of daily weather data."""

    def setUp(self):
        """Create daily data spanning 3 months."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=90, freq='D')
        self.daily = pd.DataFrame({
            'date': dates,
            'avg_temp': np.random.uniform(2, 15, 90),
            'total_rain': np.random.exponential(3, 90),
            'avg_wind': np.random.uniform(8, 30, 90),
            'avg_humidity': np.random.uniform(65, 95, 90),
            'is_rainy': np.random.choice([True, False], 90),
            'avg_severity': np.random.uniform(1, 8, 90),
            'max_severity': np.random.uniform(3, 10, 90),
        })

    def test_output_has_three_months(self):
        monthly = extract_monthly_aggregates(self.daily)
        self.assertEqual(len(monthly), 3)

    def test_season_column_exists(self):
        monthly = extract_monthly_aggregates(self.daily)
        self.assertIn('season', monthly.columns)
        self.assertEqual(monthly.iloc[0]['season'], 'Winter')  # January


class TestExtractTransportFeatures(unittest.TestCase):
    """Unit tests for transport feature extraction from CSO data."""

    def test_features_added(self):
        bus = pd.DataFrame({
            'year': [2019, 2019, 2019], 'month': [1, 2, 3],
            'passengers': [10000, 11000, 12000]
        })
        luas = pd.DataFrame({
            'year': [2019, 2019, 2019], 'month': [1, 2, 3],
            'passengers': [4000, 4500, 5000]
        })
        bus_feat, luas_feat = extract_transport_features(bus, luas)
        for feat_col in ['season', 'is_covid', 'passengers_norm', 'rolling_avg_3m']:
            self.assertIn(feat_col, bus_feat.columns, f"Bus missing: {feat_col}")
            self.assertIn(feat_col, luas_feat.columns, f"Luas missing: {feat_col}")

    def test_covid_flag(self):
        bus = pd.DataFrame({
            'year': [2019, 2020, 2021, 2022], 'month': [6, 6, 6, 6],
            'passengers': [10000, 5000, 6000, 9000]
        })
        luas = bus.copy()
        bus_feat, _ = extract_transport_features(bus, luas)
        self.assertFalse(bus_feat.iloc[0]['is_covid'])
        self.assertTrue(bus_feat.iloc[1]['is_covid'])
        self.assertTrue(bus_feat.iloc[2]['is_covid'])
        self.assertFalse(bus_feat.iloc[3]['is_covid'])


class TestExtractMergedFeatures(unittest.TestCase):
    """Unit tests for merged (weather × transport) feature extraction."""

    def test_features_added(self):
        np.random.seed(42)
        merged = pd.DataFrame({
            'year': list(range(2018, 2024)) * 2,
            'month': [1] * 6 + [7] * 6,
            'bus_passengers': np.random.randint(8000, 15000, 12),
            'luas_passengers': np.random.randint(3000, 8000, 12),
            'total_rain': np.random.uniform(20, 120, 12),
            'mean_temp': np.random.uniform(3, 18, 12),
            'avg_severity': np.random.uniform(2, 7, 12),
            'season': ['Winter'] * 6 + ['Summer'] * 6,
        })
        result = extract_merged_features(merged)
        for col in ['rain_group', 'temp_group', 'severity_group',
                     'total_passengers', 'bus_share', 'weather_impact_score']:
            self.assertIn(col, result.columns, f"Missing: {col}")

    def test_total_passengers_sum(self):
        """Need 3+ distinct values per numeric column for qcut to work."""
        merged = pd.DataFrame({
            'year': [2022, 2022, 2022], 'month': [1, 6, 11],
            'bus_passengers': [10000, 12000, 9000],
            'luas_passengers': [5000, 6000, 4000],
            'total_rain': [30.0, 50.0, 90.0],
            'mean_temp': [5.0, 15.0, 10.0],
            'avg_severity': [6.0, 3.0, 4.5],
            'season': ['Winter', 'Summer', 'Autumn'],
        })
        result = extract_merged_features(merged)
        # Row index 0: bus=10000, luas=5000 → total=15000
        self.assertEqual(result.iloc[0]['total_passengers'], 15000)
        self.assertAlmostEqual(result.iloc[0]['bus_share'], 10000 / 15000, places=3)


class TestIntegrationPipeline(unittest.TestCase):
    """
    Integration test: verifies full pipeline from data → transform → DB → query.
    Tests frontend (query/retrieval) and backend (storage) interaction.
    """

    def setUp(self):
        # Use tempfile for cross-platform compatibility (Windows + Linux)
        fd, self.test_db = tempfile.mkstemp(suffix='.db')
        os.close(fd)

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_full_pipeline_roundtrip(self):
        """End-to-end: create data → extract features → transform → store → query back."""

        # Step 1: Simulate raw hourly weather data (mimicking Open-Meteo API)
        np.random.seed(42)
        raw_weather = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=90 * 24, freq='h'),
            'temp_c': np.random.uniform(-2, 18, 90 * 24),
            'humidity': np.random.uniform(50, 100, 90 * 24),
            'rain_mm': np.random.exponential(1.5, 90 * 24),
            'wind_speed_kmh': np.random.uniform(5, 35, 90 * 24),
            'pressure_hpa': np.random.uniform(990, 1025, 90 * 24),
            'cloud_cover': np.random.uniform(0, 100, 90 * 24),
            'source': 'open_meteo_api'
        })

        # Step 2: Extract temporal + weather features
        raw_weather = extract_temporal_features(raw_weather, 'timestamp')
        raw_weather = extract_weather_features(raw_weather)

        # Step 3: Validate feature extraction
        self.assertIn('rain_category', raw_weather.columns)
        self.assertIn('weather_severity', raw_weather.columns)
        self.assertIn('season', raw_weather.columns)
        self.assertIn('comfort_index', raw_weather.columns)
        self.assertIn('is_rush_hour', raw_weather.columns)

        # Jan-Feb rows should be Winter
        jan_feb = raw_weather[raw_weather['month'].isin([1, 2])]
        self.assertTrue(all(jan_feb['season'] == 'Winter'))

        # Step 4: Aggregate to daily and monthly
        daily = extract_daily_aggregates(raw_weather)
        monthly = extract_monthly_aggregates(daily)
        self.assertEqual(len(daily), 90)
        self.assertEqual(len(monthly), 3)  # Jan, Feb, Mar

        # Step 5: Create simulated transport data (mimicking CSO API)
        bus_df = pd.DataFrame({
            'year': [2023, 2023, 2023], 'month': [1, 2, 3],
            'passengers': [12000, 11500, 12500]
        })
        luas_df = pd.DataFrame({
            'year': [2023, 2023, 2023], 'month': [1, 2, 3],
            'passengers': [5000, 4800, 5200]
        })
        bus_feat, luas_feat = extract_transport_features(bus_df, luas_df)
        self.assertIn('passengers_norm', bus_feat.columns)
        self.assertIn('is_covid', luas_feat.columns)

        # Step 6: Merge weather + transport
        merged = monthly.merge(
            bus_feat[['year', 'month', 'passengers']].rename(
                columns={'passengers': 'bus_passengers'}),
            on=['year', 'month'], how='inner'
        )
        merged = merged.merge(
            luas_feat[['year', 'month', 'passengers']].rename(
                columns={'passengers': 'luas_passengers'}),
            on=['year', 'month'], how='inner'
        )
        merged = extract_merged_features(merged)
        self.assertIn('total_passengers', merged.columns)
        self.assertIn('weather_impact_score', merged.columns)

        # Step 7: Store in database
        conn = sqlite3.connect(self.test_db)
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
                date TEXT, avg_temp REAL, max_temp REAL, min_temp REAL,
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
                max_severity REAL, severe_days INTEGER, season TEXT
            );
            CREATE TABLE bus_passengers (
                year INTEGER, month INTEGER, passengers INTEGER,
                passengers_norm REAL, yoy_change REAL, rolling_avg_3m REAL,
                season TEXT, is_covid INTEGER
            );
            CREATE TABLE luas_passengers (
                year INTEGER, month INTEGER, passengers INTEGER,
                passengers_norm REAL, yoy_change REAL, rolling_avg_3m REAL,
                season TEXT, is_covid INTEGER
            );
            CREATE TABLE weather_transport_merged (
                year INTEGER, month INTEGER,
                bus_passengers INTEGER, luas_passengers INTEGER,
                total_passengers INTEGER, bus_share REAL,
                total_rain REAL, mean_temp REAL, avg_wind REAL,
                avg_severity REAL, rainy_days REAL, severe_days INTEGER,
                season TEXT, rain_group TEXT, temp_group TEXT,
                severity_group TEXT, weather_impact_score REAL
            );
        """)
        conn.commit()

        load_to_database(conn, raw_weather, daily, monthly,
                         bus_feat, luas_feat, merged)

        # Step 8: Query from database (simulating frontend API request)
        # Daily totals (sum of 24 hourly values) are typically high, so query any non-dry category
        result = pd.read_sql_query(
            "SELECT * FROM weather_daily WHERE rain_category != 'dry'", conn)
        self.assertGreater(len(result), 0, 'Should find rainy days')

        # Step 9: Verify aggregate query (like frontend dashboard)
        agg = pd.read_sql_query("""
            SELECT rain_category,
                   COUNT(*) as days,
                   ROUND(AVG(avg_severity), 2) as avg_severity
            FROM weather_daily
            GROUP BY rain_category
        """, conn)
        self.assertGreater(len(agg), 0, 'Aggregation should return groups')

        # Step 10: Verify merged table
        merged_db = pd.read_sql_query(
            "SELECT * FROM weather_transport_merged", conn)
        self.assertEqual(len(merged_db), 3, 'Merged table should have 3 months')
        self.assertTrue(all(merged_db['total_passengers'] > 0))

        # Step 11: Row count consistency
        hourly_count = pd.read_sql_query(
            "SELECT COUNT(*) as n FROM weather_hourly", conn).iloc[0]['n']
        self.assertEqual(hourly_count, 90 * 24, 'DB should have all hourly records')

        conn.close()


def run_tests():
    """Run all tests and return results as JSON-serialisable dict."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestClassifyRainfall,
        TestMapSeason,
        TestWeatherSeverity,
        TestNormalisePassengers,
        TestValidateDataframe,
        TestExtractTemporalFeatures,
        TestExtractWeatherFeatures,
        TestExtractDailyAggregates,
        TestExtractMonthlyAggregates,
        TestExtractTransportFeatures,
        TestExtractMergedFeatures,
        TestIntegrationPipeline
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # Run with a result collector
    result = unittest.TestResult()
    suite.run(result)

    # Format results
    tests_run = result.testsRun
    failures = []
    for test, traceback in result.failures:
        failures.append({
            'test': str(test),
            'message': traceback.split('\n')[-2] if traceback else ''
        })
    errors = []
    for test, traceback in result.errors:
        errors.append({
            'test': str(test),
            'message': traceback.split('\n')[-2] if traceback else ''
        })

    return {
        'total': tests_run,
        'passed': tests_run - len(failures) - len(errors),
        'failed': len(failures),
        'errors': len(errors),
        'failures': failures,
        'error_list': errors,
        'success': len(failures) == 0 and len(errors) == 0
    }


if __name__ == '__main__':
    unittest.main(verbosity=2)
