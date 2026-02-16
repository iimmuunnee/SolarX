"""Response schemas for API endpoints."""
from pydantic import BaseModel, Field
from typing import List, Optional
from .common import PredictionMetrics, TimeSeriesData, SimulationMetadata


class VendorInfo(BaseModel):
    """Battery vendor specifications."""

    id: str = Field(..., description="Vendor identifier (lg, samsung, tesla)")
    name: str = Field(..., description="Full vendor name")
    c_rate: float = Field(..., description="Maximum C-rate (charge/discharge rate)")
    efficiency: float = Field(..., description="Round-trip efficiency (0-1)")
    soc_range: List[float] = Field(..., description="[min_soc, max_soc] operational range")
    cost_per_kwh: float = Field(..., description="Battery cost in USD/kWh")
    degradation_rate: float = Field(..., description="Capacity degradation per cycle")
    chemistry: str = Field(default="", description="Battery chemistry type")


class VendorsResponse(BaseModel):
    """Response containing all available vendors."""

    vendors: List[VendorInfo] = Field(..., description="List of available battery vendors")


class VendorResult(BaseModel):
    """Simulation results for a single vendor."""

    vendor_id: str = Field(..., description="Vendor identifier")
    vendor_name: str = Field(..., description="Full vendor name")
    revenue_krw: float = Field(..., description="Total revenue in KRW")
    soh_percent: float = Field(..., description="Final State of Health in %")
    cycle_count: float = Field(..., description="Total charge/discharge cycles")
    throughput_kwh: float = Field(..., description="Total energy throughput in kWh")
    capex_krw: float = Field(..., description="Capital expenditure in KRW")
    opex_annual_krw: float = Field(..., description="Annual operating expenditure in KRW")
    roi_percent: float = Field(..., description="Return on investment in %")
    payback_years: float = Field(..., description="Payback period in years")
    npv_krw: float = Field(..., description="Net present value in KRW")


class BaselineResult(BaseModel):
    """Simulation results without ESS (baseline)."""

    revenue_krw: float = Field(..., description="Total revenue without battery in KRW")


class BenchmarkResponse(BaseModel):
    """Response for benchmark simulation comparing all vendors."""

    simulation_id: str = Field(..., description="Unique simulation identifier")
    metadata: SimulationMetadata = Field(..., description="Simulation metadata")
    prediction_metrics: PredictionMetrics = Field(..., description="LSTM prediction quality metrics")
    vendors: List[VendorResult] = Field(..., description="Results for each vendor")
    baseline: BaselineResult = Field(..., description="Baseline results without ESS")
    time_series: TimeSeriesData = Field(..., description="Time series data for visualization")


class CustomResponse(BaseModel):
    """Response for custom simulation with specific vendor."""

    simulation_id: str = Field(..., description="Unique simulation identifier")
    metadata: SimulationMetadata = Field(..., description="Simulation metadata")
    prediction_metrics: PredictionMetrics = Field(..., description="LSTM prediction quality metrics")
    vendor_result: VendorResult = Field(..., description="Results for the selected vendor")
    baseline: BaselineResult = Field(..., description="Baseline results without ESS")
    time_series: TimeSeriesData = Field(..., description="Time series data for visualization")


class ScenarioResult(BaseModel):
    """Results for a scalability scenario."""

    name: str = Field(..., description="Scenario name (e.g., 'Donghae (Base)')")
    factor: float = Field(..., description="Solar generation scaling factor")
    vendor_id: str = Field(..., description="Vendor used for this scenario")
    revenue_krw: float = Field(..., description="Total revenue in KRW")
    soh_percent: float = Field(..., description="Final State of Health in %")
    time_series: TimeSeriesData = Field(..., description="Time series data for this scenario")


class ScalabilityResponse(BaseModel):
    """Response for scalability test results."""

    scenarios: List[ScenarioResult] = Field(..., description="Results for different regional scenarios")


class HealthResponse(BaseModel):
    """Health check response."""

    model_config = {"protected_namespaces": ()}  # Allow "model_" prefix

    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether LSTM model is loaded")
    data_loaded: bool = Field(..., description="Whether training data is loaded")
