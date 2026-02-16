"""Unit tests for data loading and preprocessing."""
import pytest
import numpy as np
import pandas as pd
from src.data_loader import SolarDataManager


def test_sequence_creation_correct_shape():
    """Test that sequence creation produces correct shapes."""
    loader = SolarDataManager()

    data_x = np.random.rand(100, 8)
    data_y = np.random.rand(100, 1)
    seq_length = 24

    X, y = loader.create_sequences(data_x, data_y, seq_length=seq_length)

    # Check shapes
    expected_samples = 100 - seq_length
    assert X.shape == (expected_samples, seq_length, 8), f"X shape mismatch: {X.shape}"
    assert y.shape == (expected_samples, 1), f"y shape mismatch: {y.shape}"


def test_scaler_consistency():
    """Test that scaler transforms are consistent."""
    loader = SolarDataManager()

    # Create sample data
    data = np.random.rand(100, 1) * 1000

    # Fit scaler
    loader.scaler_y.fit(data)

    # Transform and inverse should be consistent
    scaled = loader.scaler_y.transform(data)
    inversed = loader.inverse_transform_y(scaled)

    np.testing.assert_array_almost_equal(data, inversed, decimal=3,
                                          err_msg="Inverse transform did not recover original data")


def test_sequence_temporal_order():
    """Test that sequences preserve temporal order."""
    loader = SolarDataManager()

    # Create sequential data
    data_x = np.arange(100).reshape(-1, 1).repeat(8, axis=1)
    data_y = np.arange(100).reshape(-1, 1)
    seq_length = 24

    X, y = loader.create_sequences(data_x, data_y, seq_length=seq_length)

    # Check first sequence
    first_seq_start = X[0, 0, 0]
    first_seq_end = X[0, -1, 0]

    assert first_seq_start == 0, "First sequence should start at 0"
    assert first_seq_end == seq_length - 1, f"First sequence should end at {seq_length-1}"

    # Check that target is next timestep
    assert y[0, 0] == seq_length, f"Target should be {seq_length}"


def test_empty_data_handling():
    """Test handling of empty or insufficient data."""
    loader = SolarDataManager()

    # Test with data smaller than sequence length
    data_x = np.random.rand(10, 8)
    data_y = np.random.rand(10, 1)
    seq_length = 24

    X, y = loader.create_sequences(data_x, data_y, seq_length=seq_length)

    # Should return empty arrays
    assert len(X) == 0, "Should return empty array for insufficient data"
    assert len(y) == 0, "Should return empty array for insufficient data"


def test_weather_column_normalization():
    """Test that weather column names are normalized correctly."""
    loader = SolarDataManager()

    # Create DataFrame with various column name formats
    df = pd.DataFrame({
        "기온(°C)": [20, 21],
        "강수량": [0, 1],
        "풍속": [2, 3],
    })

    normalized = loader._normalize_weather_columns(df)

    assert "기온(℃)" in normalized.columns
    assert "강수량(mm)" in normalized.columns
    assert "풍속(m/s)" in normalized.columns
