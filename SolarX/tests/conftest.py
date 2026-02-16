"""Pytest fixtures for SolarX tests."""
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.battery import ESSBattery, LGEnergySolution, SamsungSDI, TeslaBattery
from src.model import SolarLSTM


@pytest.fixture
def sample_weather_data():
    """Generate synthetic weather data for testing."""
    hours = 48
    return pd.DataFrame({
        "Datetime": pd.date_range("2024-01-01", periods=hours, freq="h"),
        "기온(℃)": np.random.uniform(10, 25, hours),
        "강수량(mm)": np.random.uniform(0, 5, hours),
        "풍속(m/s)": np.random.uniform(0, 10, hours),
        "습도(%)": np.random.uniform(40, 80, hours),
        "일조(hr)": np.random.uniform(0, 1, hours),
        "일사(MJ/m2)": np.random.uniform(0, 3, hours),
        "운량(10분위)": np.random.randint(0, 10, hours),
        "발전량": np.random.uniform(0, 5000, hours)  # Wh
    })


@pytest.fixture
def sample_battery():
    """Create a test battery instance."""
    return ESSBattery(
        name="Test Battery",
        capacity=100.0,  # kWh
        c_rate=2.0,
        eff=0.98,
        soc_range=(0.05, 0.95),
        eff_is_roundtrip=True
    )


@pytest.fixture
def lg_battery():
    """Create LG battery for testing."""
    return LGEnergySolution(capacity=100.0)


@pytest.fixture
def samsung_battery():
    """Create Samsung battery for testing."""
    return SamsungSDI(capacity=100.0)


@pytest.fixture
def tesla_battery():
    """Create Tesla battery for testing."""
    return TeslaBattery(capacity=100.0)


@pytest.fixture
def sample_lstm_model():
    """Create a lightweight LSTM model for testing."""
    return SolarLSTM(input_size=8, hidden_size=16, num_layers=1)


@pytest.fixture
def sample_sequences():
    """Generate sample input sequences for LSTM."""
    batch_size = 4
    seq_length = 24
    input_size = 8

    X = np.random.rand(batch_size, seq_length, input_size).astype(np.float32)
    y = np.random.rand(batch_size, 1).astype(np.float32)

    return X, y
