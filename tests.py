"""
tests.py — Unit Tests + Integration Test
=========================================
Weather Impact on Dublin Public Transport — ETL Pipeline
B9AI001 Programming for Data Analytics — CA2
"""

import unittest
import numpy as np
import pandas as pd
import sqlite3
import os
import json

from pipeline import (
    classify_rainfall, map_season, compute_weather_severity,
    normalise_passengers, validate_dataframe,
    extract_temporal_features, extract_weather_features,
    extract_merged_features, create_database, load_to_database,
    DB_PATH
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
    """Unit tests for temporal feature extraction."""

    def test_features_added(self):
        df = pd.DataFrame({'date': pd.date_range('2023-06-15', periods=3)})
        result = extract_temporal_features(df, 'date')
        expected_cols = ['year', 'month', 'day', 'weekday', 'day_of_week',
                         'is_weekend', 'season', 'quarter', 'month_name']
        for col in expected_cols:
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_season_mapping(self):
        df = pd.DataFrame({'date': [pd.Timestamp('2023-01-15'),
                                     pd.Timestamp('2023-07-15')]})
        result = extract_temporal_features(df, 'date')
        self.assertEqual(result.iloc[0]['season'], 'Winter')
        self.assertEqual(result.iloc[1]['season'], 'Summer')

    def test_weekend_detection(self):
        df = pd.DataFrame({'date': [pd.Timestamp('2023-06-17'),   # Saturday
                                     pd.Timestamp('2023-06-19')]}) # Monday
        result = extract_temporal_features(df, 'date')
        self.assertTrue(result.iloc[0]['is_weekend'])
        self.assertFalse(result.iloc[1]['is_weekend'])


class TestExtractWeatherFeatures(unittest.TestCase):
    """Unit tests for weather feature extraction."""

    def test_features_added(self):
        df = pd.DataFrame({
            'maxtp': [15.0, 20.0],
            'mintp': [5.0, 10.0],
            'rain': [0.0, 10.0],
            'wdsp': [10.0, 20.0]
        })
        result = extract_weather_features(df)
        self.assertIn('avg_temp', result.columns)
        self.assertIn('temp_range', result.columns)
        self.assertIn('rain_category', result.columns)
        self.assertIn('weather_severity', result.columns)
        self.assertIn('comfort_index', result.columns)

    def test_avg_temp_calculation(self):
        df = pd.DataFrame({'maxtp': [20.0], 'mintp': [10.0], 'rain': [0], 'wdsp': [0]})
        result = extract_weather_features(df)
        self.assertAlmostEqual(result.iloc[0]['avg_temp'], 15.0)

    def test_temp_range(self):
        df = pd.DataFrame({'maxtp': [20.0], 'mintp': [5.0], 'rain': [0], 'wdsp': [0]})
        result = extract_weather_features(df)
        self.assertAlmostEqual(result.iloc[0]['temp_range'], 15.0)


class TestIntegrationPipeline(unittest.TestCase):
    """
    Integration test: verifies full pipeline from data → transform → DB → query.
    This tests frontend (query/retrieval) and backend (storage) interaction.
    """

    def setUp(self):
        self.test_db = os.path.join('/tmp', 'test_integration.db')

    def tearDown(self):
        if os.path.exists(self.test_db):
            os.remove(self.test_db)

    def test_full_pipeline_roundtrip(self):
        """End-to-end: create data → extract features → transform → store → query back."""

        # Step 1: Simulate raw weather data (mimicking API acquisition)
        raw_weather = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=90),
            'maxtp': np.random.uniform(5, 18, 90),
            'mintp': np.random.uniform(-2, 8, 90),
            'rain': np.random.exponential(3, 90),
            'wdsp': np.random.uniform(5, 25, 90),
            'sun': np.random.uniform(0, 8, 90)
        })

        # Step 2: Extract features
        raw_weather = extract_temporal_features(raw_weather, 'date')
        raw_weather = extract_weather_features(raw_weather)

        # Step 3: Validate feature extraction
        self.assertIn('rain_category', raw_weather.columns)
        self.assertIn('weather_severity', raw_weather.columns)
        self.assertIn('season', raw_weather.columns)
        self.assertIn('comfort_index', raw_weather.columns)
        # Jan-Feb rows should be Winter
        jan_feb = raw_weather[raw_weather['month'].isin([1, 2])]
        self.assertTrue(all(jan_feb['season'] == 'Winter'))

        # Step 4: Store in database
        conn = sqlite3.connect(self.test_db)
        store_df = raw_weather.copy()
        store_df['date'] = store_df['date'].astype(str)
        if 'is_rainy' in store_df.columns:
            store_df['is_rainy'] = store_df['is_rainy'].astype(int)
        if 'is_weekend' in store_df.columns:
            store_df['is_weekend'] = store_df['is_weekend'].astype(int)
        store_df.to_sql('weather_daily', conn, if_exists='replace', index=False)

        # Step 5: Query from database (simulating frontend API request)
        result = pd.read_sql_query(
            "SELECT * FROM weather_daily WHERE rain_category = 'moderate'", conn)

        # Step 6: Validate round-trip integrity
        self.assertGreater(len(result), 0, 'Should find moderate rain days')
        for _, row in result.iterrows():
            self.assertGreater(row['rain'], 5.0)
            self.assertLessEqual(row['rain'], 15.0)

        # Step 7: Verify aggregate query (like frontend dashboard would use)
        agg = pd.read_sql_query("""
            SELECT rain_category,
                   COUNT(*) as days,
                   ROUND(AVG(weather_severity), 2) as avg_severity
            FROM weather_daily
            GROUP BY rain_category
        """, conn)
        self.assertGreater(len(agg), 0, 'Aggregation should return groups')

        # Step 8: Verify row count consistency
        db_count = pd.read_sql_query(
            "SELECT COUNT(*) as n FROM weather_daily", conn).iloc[0]['n']
        self.assertEqual(db_count, 90, 'DB should have all 90 records')

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
