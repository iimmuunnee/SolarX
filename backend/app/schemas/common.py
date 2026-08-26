"""Common Pydantic schemas used across the API."""
from pydantic import BaseModel, Field
from typing import List


class PredictionMetrics(BaseModel):
    """Metrics for LSTM prediction quality."""

    mae_kw: float = Field(..., description="Mean Absolute Error in kW")
    rmse_kw: float = Field(..., description="Root Mean Squared Error in kW")
    mape_percent: float = Field(..., description="Mean Absolute Percentage Error in %")


class TimeSeriesData(BaseModel):
    """Time series data for visualization."""

    hours: List[float] = Field(..., description="Time steps in hours")
    actual_generation_kw: List[float] = Field(..., description="Actual solar generation in kW")
    predicted_generation_kw: List[float] = Field(..., description="Predicted solar generation in kW")
    lg_profit_krw: List[float] = Field(default=None, description="Cumulative profit for LG battery in KRW")
    samsung_profit_krw: List[float] = Field(default=None, description="Cumulative profit for Samsung battery in KRW")
    tesla_profit_krw: List[float] = Field(default=None, description="Cumulative profit for Tesla battery in KRW")
    baseline_profit_krw: List[float] = Field(..., description="Cumulative profit without ESS (baseline) in KRW")
    # Hourly State-of-Charge (%) of the representative vendor (Samsung, the top
    # earner) and the hourly SMP price used for decisions. Both are already
    # computed during simulation; exposing them lets the frontend visualize the
    # actual charge/discharge behavior. Optional for backward compatibility with
    # older cached/fallback responses that predate these fields.
    soc_percent: List[float] = Field(default=None, description="Hourly State of Charge (%) of the representative vendor")
    smp_price_krw: List[float] = Field(default=None, description="Hourly SMP price in KRW/kWh used for charge/discharge decisions")


class SimulationMetadata(BaseModel):
    """Metadata about the simulation run."""

    duration_hours: int = Field(..., description="Total simulation duration in hours")
    simulation_years: float = Field(..., description="Simulation duration in years")
    avg_smp_price: float = Field(..., description="Average SMP price in KRW/kWh")
    data_points: int = Field(..., description="Number of data points simulated")
