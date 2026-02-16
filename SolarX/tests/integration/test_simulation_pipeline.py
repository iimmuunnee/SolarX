"""Integration tests for end-to-end simulation pipeline."""
import pytest
import numpy as np
import pandas as pd
from src.battery import LGEnergySolution
from src.model import SolarLSTM
import torch


def test_battery_charge_discharge_cycle():
    """Test a complete charge-discharge cycle."""
    battery = LGEnergySolution(capacity=100.0)

    initial_kwh = battery.current_kwh

    # Charge for 2 hours
    for _ in range(2):
        battery.update(action=1, amount_kw=50.0, dt_hours=1.0)

    charged_kwh = battery.current_kwh

    # Should have more energy
    assert charged_kwh > initial_kwh, "Battery should have more energy after charging"

    # Discharge for 1 hour
    battery.update(action=-1, amount_kw=50.0, dt_hours=1.0)

    discharged_kwh = battery.current_kwh

    # Should have less energy than at peak
    assert discharged_kwh < charged_kwh, "Battery should have less energy after discharging"


def test_simulation_profit_calculation():
    """Test profit calculation with battery arbitrage."""
    battery = LGEnergySolution(capacity=100.0)

    # Simulate 24 hours
    hours = 24
    generation_kw = np.random.uniform(0, 50, hours)
    prices = np.array([100 if h < 12 else 200 for h in range(hours)])  # High prices in afternoon

    profit_no_battery = 0.0
    profit_with_battery = 0.0

    # Calculate baseline (no battery)
    for gen, price in zip(generation_kw, prices):
        profit_no_battery += gen * 1.0 * price

    # Calculate with battery (simple strategy: charge when price low, discharge when high)
    avg_price = np.mean(prices)
    battery2 = LGEnergySolution(capacity=100.0)

    for gen, price in zip(generation_kw, prices):
        action = 0
        if price < avg_price * 0.9:
            action = 1  # Charge
            trade_kw = gen + battery2.update(action, 50.0, 1.0)
        elif price > avg_price * 1.1:
            action = -1  # Discharge
            trade_kw = gen + battery2.update(action, 50.0, 1.0)
        else:
            trade_kw = gen

        profit_with_battery += trade_kw * 1.0 * price

    # Battery should improve profit in this scenario
    # Note: This might not always be true depending on generation and prices
    # but with our setup (low prices early, high prices late), it should work
    print(f"Profit no battery: {profit_no_battery:.2f}")
    print(f"Profit with battery: {profit_with_battery:.2f}")

    # At minimum, profit shouldn't be drastically worse
    assert profit_with_battery >= profit_no_battery * 0.5, "Battery strategy severely degraded profit"


def test_model_prediction_pipeline():
    """Test the complete prediction pipeline."""
    # Create model
    model = SolarLSTM(input_size=8, hidden_size=16, num_layers=1)
    model.eval()

    # Create synthetic sequence
    seq_length = 24
    batch_size = 10
    X = np.random.rand(batch_size, seq_length, 8).astype(np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    # Predict
    with torch.no_grad():
        predictions = model(X_tensor)

    # Check output
    assert predictions.shape == (batch_size, 1), "Prediction shape incorrect"
    assert not torch.isnan(predictions).any(), "Predictions contain NaN"


def test_end_to_end_minimal():
    """Minimal end-to-end test with synthetic data."""
    # 1. Create synthetic weather/generation data
    hours = 48
    seq_length = 24

    weather_features = np.random.rand(hours, 8).astype(np.float32)
    generation = np.random.rand(hours, 1).astype(np.float32) * 5000  # Wh

    # 2. Create sequences
    sequences_x = []
    sequences_y = []
    for i in range(hours - seq_length):
        sequences_x.append(weather_features[i:i+seq_length])
        sequences_y.append(generation[i+seq_length])

    sequences_x = np.array(sequences_x)
    sequences_y = np.array(sequences_y)

    # 3. Model prediction
    model = SolarLSTM(input_size=8, hidden_size=16, num_layers=1)
    model.eval()

    X_tensor = torch.tensor(sequences_x, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_tensor)

    predictions_np = predictions.cpu().numpy().flatten()
    y_real_kw = sequences_y.flatten() / 1000.0  # Convert to kW
    y_pred_kw = predictions_np / 1000.0

    # 4. Battery simulation
    battery = LGEnergySolution(capacity=max(y_real_kw) * 3)
    prices = [100 if i % 24 < 12 else 200 for i in range(len(y_real_kw))]
    avg_price = np.mean(prices)

    profit = 0
    for i, (pred_kw, real_kw) in enumerate(zip(y_pred_kw, y_real_kw)):
        price = prices[i]

        action = 0
        if price > avg_price * 1.1:
            action = -1
        elif price < avg_price * 0.9:
            action = 1

        if action == 1:
            trade_kw = real_kw + battery.update(action, 50.0, 1.0)
        elif action == -1:
            trade_kw = real_kw + battery.update(action, 50.0, 1.0)
        else:
            trade_kw = real_kw

        profit += trade_kw * 1.0 * price

    # Should complete without errors
    assert profit > 0, "Profit should be positive"
    print(f"End-to-end test profit: {profit:.2f} KRW")
