"""Unit tests for battery physics and constraints."""
import pytest
import numpy as np


def test_charge_within_crate_limit(sample_battery):
    """Test that charging respects C-rate limit."""
    battery = sample_battery
    initial_soc = battery.current_kwh / battery.capacity

    # Request more than C-rate allows
    requested_power = battery.max_power * 2
    grid_power = battery.update(action=1, amount_kw=requested_power, dt_hours=1.0)

    # Should be clamped to max_power
    assert abs(grid_power) <= battery.max_power, "Charge power exceeded C-rate limit"


def test_discharge_respects_soc_min(sample_battery):
    """Test that discharge respects SoC minimum."""
    battery = sample_battery

    # Discharge everything
    for _ in range(20):
        battery.update(action=-1, amount_kw=battery.max_power, dt_hours=1.0)

    # SoC should not go below minimum
    current_soc = battery.current_kwh / battery.capacity
    assert current_soc >= battery.soc_min - 1e-6, f"SoC {current_soc} below minimum {battery.soc_min}"


def test_efficiency_energy_conservation(sample_battery):
    """Test energy conservation with efficiency."""
    battery = sample_battery
    initial_kwh = battery.current_kwh

    # Charge 10 kWh
    charge_kwh = 10.0
    battery.update(action=1, amount_kw=charge_kwh, dt_hours=1.0)

    # Energy gained should be less than input due to efficiency
    energy_gained = battery.current_kwh - initial_kwh
    assert energy_gained < charge_kwh, "Energy gained should be less than input due to efficiency"
    assert energy_gained > 0, "Some energy should be stored"


def test_cannot_discharge_more_than_available(sample_battery):
    """Test that discharge cannot exceed available energy."""
    battery = sample_battery
    initial_kwh = battery.current_kwh

    # Request massive discharge
    battery.update(action=-1, amount_kw=battery.max_power * 10, dt_hours=1.0)

    # Battery should not go negative
    assert battery.current_kwh >= battery.capacity * battery.soc_min, "Battery discharged below minimum"


def test_charge_to_soc_max(sample_battery):
    """Test that charging stops at SoC maximum."""
    battery = sample_battery

    # Charge to maximum
    for _ in range(50):
        battery.update(action=1, amount_kw=battery.max_power, dt_hours=1.0)

    # SoC should not exceed maximum
    current_soc = battery.current_kwh / battery.capacity
    assert current_soc <= battery.soc_max + 1e-6, f"SoC {current_soc} exceeded maximum {battery.soc_max}"


def test_idle_action_no_change(sample_battery):
    """Test that idle action (0) does not change battery state."""
    battery = sample_battery
    initial_kwh = battery.current_kwh

    grid_power = battery.update(action=0, amount_kw=0, dt_hours=1.0)

    assert grid_power == 0, "Idle action should not trade with grid"
    assert battery.current_kwh == initial_kwh, "Idle action should not change battery state"


def test_vendor_specifications(lg_battery, samsung_battery, tesla_battery):
    """Test that vendor-specific batteries have correct specifications."""
    # LG: 2.0C, 98.0%
    assert lg_battery.max_power == lg_battery.capacity * 2.0
    assert abs(lg_battery.charge_eff * lg_battery.discharge_eff - 0.98) < 1e-6

    # Samsung: 1.8C, 98.5%
    assert samsung_battery.max_power == samsung_battery.capacity * 1.8
    assert abs(samsung_battery.charge_eff * samsung_battery.discharge_eff - 0.985) < 1e-6

    # Tesla: 1.5C, 97.0%
    assert tesla_battery.max_power == tesla_battery.capacity * 1.5
    assert abs(tesla_battery.charge_eff * tesla_battery.discharge_eff - 0.97) < 1e-6


def test_round_trip_efficiency_conversion(sample_battery):
    """Test that round-trip efficiency is correctly converted to one-way."""
    battery = sample_battery
    # Round-trip = 0.98, so one-way should be sqrt(0.98)
    expected_one_way = np.sqrt(0.98)
    assert abs(battery.charge_eff - expected_one_way) < 1e-6
    assert abs(battery.discharge_eff - expected_one_way) < 1e-6
